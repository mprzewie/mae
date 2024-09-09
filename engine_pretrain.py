# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable
from math import ceil
import torch
import warnings

from torch.utils.tensorboard import SummaryWriter

import util.misc as misc
import util.lr_sched as lr_sched


from loss_func import uniformity_loss, ClsPosLoss
from models_mae import MaskedAutoencoderViT

AMP_PRECISIONS = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "none": torch.float32,
}


def train_one_epoch(model: MaskedAutoencoderViT,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    cls_pos_loss: ClsPosLoss,
                    log_writer: SummaryWriter = None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(
                enabled=args.amp != "none",
                dtype=AMP_PRECISIONS[args.amp]
        ):
            loss_mae, _, _, (cls_feats, outputs, latent, latent_proj, ids_restore, latent_pred) = model.forward(samples,
                                                                                                   mask_ratio=args.mask_ratio)

            if args.umae_reg == 'none':
                loss_reg = torch.zeros_like(loss_mae)
            else:
                loss_reg = uniformity_loss(cls_feats)

            target_latent = latent_proj[:, 1:]

            loss_latent = cls_pos_loss.forward(target_latent, latent_pred, epoch=epoch)

            outputs_ce = outputs[targets >= 0]
            targets_ce = targets[targets >= 0]

            if len(targets_ce) > 0:
                loss_ce = torch.nn.functional.cross_entropy(outputs_ce, targets_ce)
            else:
                loss_ce = torch.tensor(0.).to(device)

        loss = loss_mae + (args.lamb * loss_reg) + (args.lpred_lambda * loss_latent) + loss_ce

        loss_mae_value = loss_mae.item()
        loss_reg_value = loss_reg.item()
        loss_ce_value = loss_ce.item()
        loss_latent_value = loss_latent.item()
        loss_value = loss.item()
        train_acc = (outputs.argmax(dim=1) == targets).float().mean()

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(
            loss=loss_value,
            loss_mae=loss_mae_value,
            loss_reg=loss_reg_value,
            loss_ce=loss_ce_value,
            loss_latent=loss_latent_value,
            train_acc=train_acc
        )

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_mae_value_reduce = misc.all_reduce_mean(loss_mae_value)
        loss_reg_value_reduce = misc.all_reduce_mean(loss_reg_value)
        loss_ce_value_reduce = misc.all_reduce_mean(loss_ce_value)
        loss_latent_value_reduce = misc.all_reduce_mean(loss_latent_value)
        train_acc_reduce = misc.all_reduce_mean(train_acc)

        losses = {
            "value": loss_value_reduce,
            "mae": loss_mae_value_reduce,
            "reg": loss_reg_value_reduce,
            "ce": loss_ce_value_reduce,
        }
        assert not any([math.isnan(l) for l in losses.values()]), losses

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_mae', loss_mae_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_reg', loss_reg_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_ce', loss_ce_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_latent', loss_latent_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_acc', train_acc_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar("epoch", epoch, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
