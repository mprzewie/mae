# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

import argparse
import datetime
import json
from copy import deepcopy
from typing import Type

import numpy as np
import os
import time
from pathlib import Path

import psutil
import torch
import torch.backends.cudnn as cudnn
import torchvision
from timm.utils import accuracy
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset
import timm
import gc
# import nvidia_smi

# nvidia_smi.nvmlInit()
# assert timm.__version__ == "0.3.2" # version check
from timm.models.layers import trunc_normal_
from torchvision.datasets import STL10
from tqdm import tqdm

import util.misc as misc
from util.datasets import build_dataset_v2
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.lars import LARS
from util.crop import RandomResizedCrop

import models_vit

from engine_finetune import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('MAE cls-cls attention', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    # parser.add_argument('--epochs', default=90, type=int)
    # parser.add_argument("--aug_every", type=int, default=None)

    # parser.add_argument('--accum_iter', default=1, type=int,
    #                     help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    # Optimizer parameters
    # parser.add_argument('--weight_decay', type=float, default=0,
    #                     help='weight decay (default: 0 for linear probe following MoCo v1)')
    #
    # parser.add_argument('--lr', type=float, default=None, metavar='LR',
    #                     help='learning rate (absolute lr)')
    # parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
    #                     help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    #
    # parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
    #                     help='lower lr bound for cyclic schedulers that hit 0')
    #
    # parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
    #                     help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    # parser.add_argument('--global_pool', action='store_true')
    # parser.set_defaults(global_pool=False)
    # parser.add_argument('--cls_token', action='store_false', dest='global_pool',
    #                     help='Use class token instead of global pool for classification')
    # parser.add_argument("--n_last_layers", type=int, default=1, help="Use activations from N last layers for classification")
    # parser.add_argument("--shuffle_subsets", type=int, default=1, help="Shuffle positional tokens into N subsets during inference")
    # parser.add_argument("--agg_method", choices=["rep", "log", "t1"], default="rep", help="representations / logits / take 1 of shuffled")
    # parser.add_argument("--cls_features", choices=["cls", "pos", "both"], default="cls", help="cls token / positional tokens for classification")
    # parser.add_argument("--block_reshuffling", "--br", action="store_true", help="reshuffle pos tokens btw. blocks")

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=Path,
                        help='dataset path')
    # parser.add_argument('--nb_classes', default=1000, type=int,
    #                     help='number of the classification types')

    parser.add_argument('--output_dir', default=None,
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    # parser.add_argument('--resume', default='',
    #                     help='resume from checkpoint')

    # parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
    #                     help='start epoch')
    # parser.add_argument('--eval', action='store_true',
    #                     help='Perform evaluation only')
    # parser.add_argument('--dist_eval', action='store_true', default=False,
    #                     help='Enabling distributed evaluation (recommended during training for faster monitor')
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


    return parser


def main(args):
    # misc.init_distributed_mode(args)
    # args.aug_every = args.aug_every or args.epochs

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()

    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # linear probe: weak augmentation
    dataset_train, dataset_val = build_dataset_v2(args, is_pretrain=False)

    print(dataset_train)
    print(dataset_val)

    args.distributed = False
    args.gpu = 0
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    if (not args.distributed or (global_rank == 0)) and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        misc.maybe_setup_wandb(args.output_dir, args=args, job_type="cls_cls_attn")

        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # assert False, (len(dataset_train), len(dataset_val))
    # data_loader_train = torch.utils.data.DataLoader(
    #     dataset_train, sampler=sampler_train,
    #     # batch_size=12,
    #     # batch_size=512,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=args.pin_mem,
    #     drop_last=False,
    # )

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
        size_patch_kwargs=dict(
            img_size=args.input_size,
            patch_size=args.input_size // 16
        )

    model: models_vit.VisionTransformer = models_vit.__dict__[args.model](
        num_classes=1000,
        **size_patch_kwargs
    )
    print(model)


    if args.finetune: # and not args.eval:
        if Path(args.finetune).exists():
            print("Interpreting", args.finetune, "as path")
            checkpoint_model = torch.load(args.finetune, map_location='cpu')["model"]
        else:
            print("Interpreting", args.finetune, "as timm model")
            from timm.models.vision_transformer import _create_vision_transformer

            model_to_kwargs = {
                "vit_tiny_patch16": dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
                "vit_base_patch16": dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
                "vit_large_patch16": dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
                "vit_huge_patch16": dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16),
            }

            model_kwargs = model_to_kwargs[args.model]
            checkpoint_model = _create_vision_transformer(args.finetune, pretrained=True, **model_kwargs).state_dict()


        print("Load pre-trained checkpoint from: %s" % args.finetune)
        # checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # if args.global_pool:
        #     assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
        # else:
        assert not any([k.startswith("blocks") for k in msg.missing_keys])
        # assert set(msg.missing_keys) == {'head.weight', 'head.bias'}, msg.missing_keys

        # manually initialize fc layer: following MoCo v3



    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    # eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    # if args.lr is None:  # only base_lr is specified
    #     args.lr = args.blr * eff_batch_size / 256

    # print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    # print("actual lr: %.2e" % args.lr)
    #
    # print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    cls_cls_attns = []
    with torch.no_grad():
        for (data, target) in tqdm(data_loader_val, desc="collecting features"):
            _, attns = model.forward_features(data.to(device))

            attns = torch.stack(attns)

            cls_cls_attn = attns[:, :, :, 0, 0].detach().cpu().numpy() # blocks, batch, heads
            cls_cls_attns.append(cls_cls_attn)

            # assert False, [a.shape for a in attns]

    cc_attns = np.concatenate(cls_cls_attns, axis=1)
    cc_attns = cc_attns.mean(axis=(1, 2))

    for i, a in enumerate(cc_attns):
        log_writer.add_scalar("test/cls_cls_attention", a, global_step=i)


    # for i, a in enumerate(cc_attns):
    #     writer.add_scalar("eval_attention/cls_cls", a, global_step=i)

    # optimizer = LARS(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # print(optimizer)
    # loss_scaler = NativeScaler()
    #
    # criterion = torch.nn.CrossEntropyLoss()
    #
    # print("criterion = %s" % str(criterion))

    # misc.load_model(args=args, model_without_ddp=model_without_ddp)
    # assert False, "ok"






if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
