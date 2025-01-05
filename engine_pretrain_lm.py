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
from engine_pretrain import AMP_PRECISIONS

from loss_func import uniformity_loss, ClsPosLoss
from models_latent_mixer import LatentMixer
from models_mae import MaskedAutoencoderViT

def train_lm_one_epoch(
        model: LatentMixer,

                       data_loader: Iterable, optimizer: torch.optim.Optimizer,
                       device: torch.device, epoch: int, loss_scaler,
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
            loss_disc, (acc_disc, encoder_tokens, lin_cls_outputs) = model.forward(samples)


            outputs_ce = lin_cls_outputs[targets >= 0]
            targets_ce = targets[targets >= 0]


            if len(targets_ce) > 0:
                loss_ce = torch.nn.functional.cross_entropy(outputs_ce, targets_ce)
            else:
                loss_ce = torch.tensor(0.).to(device)

        loss = loss_disc + loss_ce

        acc_disc_value = acc_disc.item()
        loss_disc_value = loss_disc.item()
        loss_ce_value = loss_ce.item()
        loss_value = loss.item()
        train_acc = (lin_cls_outputs.argmax(dim=1) == targets).float().mean()

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(
            loss=loss_value,
            loss_disc=loss_disc_value,
            loss_ce=loss_ce_value,
            disc_acc=acc_disc_value,
            cls_acc=train_acc
        )

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_disc_value_reduce = misc.all_reduce_mean(loss_disc_value)
        loss_ce_value_reduce = misc.all_reduce_mean(loss_ce_value)
        disc_acc_reduce = misc.all_reduce_mean(acc_disc_value)
        train_acc_reduce = misc.all_reduce_mean(train_acc)

        losses = {
            "value": loss_value_reduce,
            "disc": loss_disc_value_reduce,
            "ce": loss_ce_value_reduce,
        }
        assert not any([math.isnan(l) for l in losses.values()]), losses

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_disc', loss_disc_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_loss_ce', loss_ce_value_reduce, epoch_1000x)
            log_writer.add_scalar('train_acc', train_acc_reduce, epoch_1000x)
            log_writer.add_scalar('disc_acc', disc_acc_reduce, epoch_1000x),
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalar("epoch", epoch, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
