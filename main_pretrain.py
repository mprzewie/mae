# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
from typing import Type

import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

import timm.optim.optim_factory as optim_factory
from torchvision.datasets import STL10

import util.misc as misc
from loss_func import ClsPosLoss
from util.datasets import build_dataset_v2
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch, AMP_PRECISIONS
from engine_finetune import evaluate, calculate_effrank, draw_mae_predictions


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=800, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', "-npl", action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=Path, help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # new
    parser.add_argument('--lamb', type=float, default=0)
    parser.add_argument('--umae_reg', type=str, default='none', choices=['none', 'spectral'])
    parser.add_argument("--lpred_loss", type=str, default="mse", choices=["mse", "cos", "dino"])
    parser.add_argument("--lpred_lambda", type=float, default=0., help="weight of loss of latent prediction from cls token")
    # parser.add_argument("--lpred_no_detach", "-llndt", action="store_true", default=False, help="detach encoder tokens for latent prediction loss")
    parser.add_argument("--latent_loss_detach_cls", "-lldc", action="store_true", default=False)
    parser.add_argument("--lpred_decoder_arch", "-lda", choices=["vit", "mlp"], default="vit")
    parser.add_argument("--lpred_decoder_depth", type=int, default=8)
    parser.add_argument("--lpred_decoder_heads", type=int, default=16)
    parser.add_argument("--lpred_decoder_embed_dim", "-lded", type=int, default=None)

    parser.add_argument("--latent_cls_input", "-lci", choices=["cls", "pos"], default="cls")
    parser.add_argument("--latent_loss_norm_targets", "-llnt", action="store_true", default=False)

    parser.add_argument('--val_interval', default=10, type=int)
    parser.add_argument('--save_interval', default=50, type=int)
    parser.add_argument("--amp", default="float16", choices=list(AMP_PRECISIONS.keys()), type=str)

    return parser


def main(args):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    dataset_train, dataset_val = build_dataset_v2(args, is_pretrain=True)


    print(dataset_train)
    print(dataset_val)

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))

        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_val = %s" % str(sampler_val))

    else:
        num_tasks = 1
        global_rank = 0
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.RandomSampler(dataset_val)

    eff_batch_size = args.batch_size * args.accum_iter * num_tasks
    args.eff_batch_size = eff_batch_size

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        misc.maybe_setup_wandb(args.log_dir, args=args, job_type="pretrain")
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    size_patch_kwargs = dict()
    if args.input_size != 224:
        assert args.input_size % 16 == 0, args.input_size
        size_patch_kwargs = dict(
            img_size=args.input_size,
            patch_size=args.input_size // 16
        )

    # define the model
    model = models_mae.__dict__[args.model](
        norm_pix_loss=args.norm_pix_loss,
        latent_decoder_arch=args.lpred_decoder_arch,
        latent_decoder_depth=args.lpred_decoder_depth,
        latent_decoder_heads=args.lpred_decoder_heads,
        latent_loss_detach_classifier=args.latent_loss_detach_cls,
        latent_cls_input=args.latent_cls_input,
        latent_decoder_embed_dim=args.lpred_decoder_embed_dim,
        **size_patch_kwargs
    )

    model.to(device)

    model_without_ddp: models_mae.MaskedAutoencoderViT = model
    print("Model = %s" % str(model_without_ddp))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    # param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()
    cls_pos_loss = ClsPosLoss(
        args.lpred_loss,
        out_dim=model_without_ddp.embed_dim,
        warmup_teacher_temp_epochs=args.warmup_epochs,
        number_of_epochs=args.epochs,
        norm_targets=args.latent_loss_norm_targets
    )
    cls_pos_loss.to(device)


    test_stats = misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.resume:
        assert test_stats is not None


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            cls_pos_loss=cls_pos_loss,
            log_writer=log_writer,
            args=args
        )

        if epoch % args.val_interval == 0:
            test_stats = evaluate(data_loader_val, model, device)
            effrank = calculate_effrank(data_loader_val, model_without_ddp, device)

            print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f'Max accuracy: {max_accuracy:.2f}%')
            print(f"Effective rank: {effrank:.2f}")
            mae_pred = draw_mae_predictions(dataset_val, model_without_ddp, device)

            if log_writer is not None:
                log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
                log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
                log_writer.add_scalar('monitoring/effrank', effrank, epoch)
                log_writer.add_image('monitoring/mae_image', (mae_pred + 1) / 2, global_step=epoch)



        if args.output_dir and (epoch % args.save_interval  == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch, test_stats=test_stats)




        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
