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
from typing import Iterable, Optional, Union

import numpy as np
import torch
from einops import rearrange

from timm.data import Mixup
from timm.utils import accuracy
from tqdm import tqdm

import util.misc as misc
import util.lr_sched as lr_sched
from models_mae import MaskedAutoencoderViT
from models_vit import VisionTransformer


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
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

    for data_iter_step, (samples, targets) in tqdm(enumerate(metric_logger.log_every(data_loader, print_freq, header))):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model: Union[MaskedAutoencoderViT, VisionTransformer], device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model.forward(images)
            if type(output) is tuple:
                _, _, _, (_, output, _, _, _) = output

            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def calculate_effrank(data_loader, model: MaskedAutoencoderViT, device):
    Xs = []
    for val_img, _ in data_loader:
        val_img = val_img.to(device)
        latent, mask, ids_restore, (x_blocks, attn) = model.forward_encoder(val_img, mask_ratio=0)
        cls_features = latent[:, 0]
        Xs.append(cls_features.detach().cpu().numpy())

    Xs = np.concatenate(Xs, axis=0)
    U, D, V = np.linalg.svd(Xs)
    D_l1 = np.abs(D).sum()
    P = D / D_l1
    H = P * np.log(P)
    effrank = np.exp(-H.sum())
    return effrank


@torch.no_grad()
def draw_mae_predictions(dataset, model: MaskedAutoencoderViT, device):
    val_img = torch.stack([dataset[i][0] for i in range(16)]).to(device)
    mae_loss, pred, mask, (cls_feats, outputs, latent, ids_restore, latent_pred) = model.forward(val_img)
    pred_img = model.unpatchify(pred)

    d_input = torch.cat([latent[:, :1], latent_pred], 1)

    latent_pred_img_p = model.forward_decoder(d_input, ids_restore)
    latent_pred_img = model.unpatchify(latent_pred_img_p)

    patched_img = model.patchify(val_img)
    masked_patched_img = patched_img * (mask.unsqueeze(2) - 1) * (-1)
    masked_img = model.unpatchify(masked_patched_img)
    img = torch.cat([val_img, masked_img, pred_img, latent_pred_img], dim=0)
    img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=4)
    return img
