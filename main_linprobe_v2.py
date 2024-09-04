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

import matplotlib.pyplot as plt
import numpy as np
import os
import time
from pathlib import Path

import psutil
import torch
import torch.backends.cudnn as cudnn
import torchvision
import wandb
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
    parser = argparse.ArgumentParser('MAE linear probing for image classification', add_help=False)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument("--aug_every", type=int, default=None)

    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='vit_base_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay (default: 0 for linear probe following MoCo v1)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=0.1, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument("--dino_aug", action="store_true", default=False)

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=10, metavar='N',
                        help='epochs to warmup LR')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument("--checkpoint_key", default="model", type=str)
    parser.add_argument("--cca_bias", default="none")

    # parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=False)
    # parser.add_argument('--cls_token', action='store_false', dest='global_pool',
    #                     help='Use class token instead of global pool for classification')
    parser.add_argument("--n_last_layers", type=int, default=1, help="Use activations from N last layers for classification")
    parser.add_argument("--shuffle_subsets", type=int, default=1, help="Shuffle positional tokens into N subsets during inference")
    parser.add_argument("--agg_method", choices=["rep", "log", "t1"], default="rep", help="representations / logits / take 1 of shuffled")
    parser.add_argument("--cls_features", choices=["cls", "pos", "both"], default="cls", help="cls token / positional tokens for classification")
    parser.add_argument("--block_reshuffling", "--br", action="store_true", help="reshuffle pos tokens btw. blocks")

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=Path,
                        help='dataset path')
    parser.add_argument('--nb_classes', default=1000, type=int,
                        help='number of the classification types')

    parser.add_argument('--output_dir', default=None,
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
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
    parser.add_argument("--attn_only", action="store_true", default=False)


    return parser


def main(args):
    # misc.init_distributed_mode(args)
    args.aug_every = args.aug_every or args.epochs

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

    if (not args.distributed or (global_rank == 0)) and args.output_dir is not None and not args.eval:
        os.makedirs(args.output_dir, exist_ok=True)
        misc.maybe_setup_wandb(args.output_dir, args=args, job_type="linprobe")

        log_writer = SummaryWriter(log_dir=args.output_dir)
    else:
        log_writer = None

    # assert False, (len(dataset_train), len(dataset_val))
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        # batch_size=12,
        # batch_size=512,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        # batch_size=12,
        # batch_size=512,
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

    model_to_kwargs = {
        "vit_tiny_patch16": dict(patch_size=16, embed_dim=192, depth=12, num_heads=12),
        "vit_small_patch16": dict(patch_size=16, embed_dim=384, depth=12, num_heads=12),
        "vit_base_patch16": dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
        "vit_large_patch16": dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
        "vit_huge_patch14": dict(patch_size=14, embed_dim=1280, depth=32, num_heads=16),
    }
    model: models_vit.VisionTransformer = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        global_pool=False, #args.global_pool,
        n_last_layers=args.n_last_layers,
        block_reshuffling=args.block_reshuffling,
        **size_patch_kwargs
    )
    print(model)

    classifier = AggHead(model.head, agg_method=args.agg_method)

    if args.finetune and not args.eval:
        if Path(args.finetune).exists():
            print("Interpreting", args.finetune, "as path")
            checkpoint_model = torch.load(args.finetune, map_location='cpu')[args.checkpoint_key]
        else:
            print("Interpreting", args.finetune, "as timm model")
            from timm.models.vision_transformer import _create_vision_transformer


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
        trunc_normal_(classifier.mlp.weight, std=0.01)

    # for linear prob only
    # assert False, model.head
    # hack: revise model's head with BN
    # model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6), model.head)
    # model.head = torch.nn.Identity()
    # freeze all but the head
    # for _, p in model.named_parameters():
    #     p.requires_grad = False
    for _, p in classifier.named_parameters():
        p.requires_grad = True

    model.to(device)
    classifier.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module


    optimizer = LARS(classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print(optimizer)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    _, _, A_test, M_test = collect_features(
        model, data_loader_val, device, shuffle_subsets=args.shuffle_subsets, tqdm_desc="attention stats",
        return_features=args.cls_features
    )

    mean_attn_stats = A_test.mean(dim=(0, 2))
    mean_magn_stats = M_test.mean(dim=0)


    cc_attns = mean_attn_stats[:, 0]
    pos_self_attns = mean_attn_stats[:, 1]
    cc_attns_adj = mean_attn_stats[:, 2]
    pos_self_attns_adj = mean_attn_stats[:, 3]
    cls_pos_attns = mean_attn_stats[:, 4] # should complement the cls cls attention
    pos_cls_attns = mean_attn_stats[:, 5]
    cls_pos_entropy = mean_attn_stats[:, 6]
    pos_pos_entropy = mean_attn_stats[:, 7]
    cls_magnitude = mean_magn_stats[:, 0]
    pos_magnitude = mean_magn_stats[:, 1]


    stats_pf = "test_attn"
    if wandb.run is not None:
        for b in range(len(cc_attns)):
            wandb.log({
                f"{stats_pf}/cls_cls_attention": cc_attns[b],
                f"{stats_pf}/pos_self_attention": pos_self_attns[b],
                f"{stats_pf}/cls_cls_attention_adj_for_cls": cc_attns_adj[b],
                f"{stats_pf}/pos_self_attention_adj_for_cls": pos_self_attns_adj[b],
                f"{stats_pf}/cls_pos_attention": cls_pos_attns[b],
                f"{stats_pf}/pos_cls_attention": pos_cls_attns[b],
                f"{stats_pf}/cls_pos_entropy": cls_pos_entropy[b],
                f"{stats_pf}/pos_pos_entropy": pos_pos_entropy[b],
                f"{stats_pf}/cls_magnitude": cls_magnitude[b],
                f"{stats_pf}/pos_magnitude": pos_magnitude[b],
                f"{stats_pf}/vit_block": b,
            })

    if args.attn_only:
        exit(0)

    _, _, A_train, _ = collect_features(
        model, data_loader_train, device, shuffle_subsets=args.shuffle_subsets, tqdm_desc="cca bias before",
        return_features=args.cls_features
    )

    cca = A_train[:, :, :, 0].mean(dim=0)
    ccs = A_train[:, :, :, 0].std(dim=0)

    cca_mean = cca.mean(dim=1)
    n_blocks = len(model_without_ddp.blocks)
    target_cca = torch.linspace((n_blocks - 1) / n_blocks, 1 / n_blocks, n_blocks)

    if args.cca_bias.startswith("linear"):
        cca_biases = target_cca.unsqueeze(1) - cca

        if "clamp_ceil" in args.cca_bias:
            cca_biases = cca_biases.clamp(max=0)

        for bi in range(n_blocks):
            model_without_ddp.blocks[bi].attn.cls_bias = cca_biases[bi].to(device)

    elif args.cca_bias != "none":
        raise NotImplementedError(args.cca_bias)

    if args.cca_bias != "none":
        _, _, A_train, _ = collect_features(
            model, data_loader_train, device, shuffle_subsets=args.shuffle_subsets, tqdm_desc="cca after",
            return_features=args.cls_features
        )

    cca2 = A_train[:, :, :, 0].mean(dim=0)
    cca_mean2 = cca2.mean(dim=1)

    fig, ax = plt.subplots(cca.shape[1], figsize=(cca.shape[0], 2 * cca.shape[1]))
    for h in range(cca.shape[1]):
        ax[h].errorbar(list(range(len(cca))), cca[:, h], yerr=ccs[:, h], color="blue", label=f"cca before")
        ax[h].plot(cca_mean, color="red", label=f"mean cca before")
        ax[h].plot(target_cca, color="green", ls="--", label=f"target cca")
        ax[h].plot(cca2[:, h], color="orange", ls="-.", label=f"cca after")
        ax[h].plot(cca_mean2, color="gray", label=f"mean cca after")
        ax[h].set_title(f"Head {h}")
        ax[h].set_xlabel("VIT Block")
        ax[h].set_ylim(-0.1, 1.2)
        ax[0].legend(ncols=5)

    if wandb.run is not None:
        wandb.log({"monitoring/cca_bias": fig})

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0

    for epoch in range(args.start_epoch, args.epochs):

        if epoch % args.aug_every == 0:
            X_train, Y_train, _, _ = collect_features(
                model, data_loader_train, device, shuffle_subsets=args.shuffle_subsets, tqdm_desc="train",
                return_features=args.cls_features
            )
            ds_train = TensorDataset(X_train, Y_train)
            dl_train = torch.utils.data.DataLoader(
                ds_train, shuffle=True,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False,
            )

            X_test, Y_test, A_test, M_test = collect_features(
                model, data_loader_val, device, shuffle_subsets=args.shuffle_subsets, tqdm_desc="val",
                return_features=args.cls_features
            )
            ds_test = TensorDataset(X_test, Y_test)
            dl_val = torch.utils.data.DataLoader(
                ds_test, shuffle=False,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            )

        if args.distributed:
            dl_train.sampler.set_epoch(epoch)
        print(f"{epoch=}")

        train_stats = train_one_epoch(
            classifier, criterion, dl_train,
            optimizer, device, epoch, loss_scaler,
            max_norm=None,
            log_writer=log_writer,
            args=args
        )

        test_stats = evaluate(dl_val, classifier, device, return_targets_and_preds=True)

        test_targets = test_stats.pop("targets")
        test_preds = test_stats.pop("preds")

        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f'Max accuracy: {max_accuracy:.2f}%')

        lin_pf = f"test_linear_{args.cls_features}"
        if log_writer is not None:
            log_writer.add_scalar(f'{lin_pf}/test_acc1', test_stats['acc1'], epoch)
            log_writer.add_scalar(f'{lin_pf}/test_acc5', test_stats['acc5'], epoch)
            log_writer.add_scalar(f'{lin_pf}/test_loss', test_stats['loss'], epoch)
            for k, v in train_stats.items():
                if isinstance(v, float):
                    log_writer.add_scalar(f"{lin_pf}/train_{k}", v, epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    outputs = {
        "targets": test_targets,
        "preds": test_preds,
        "attentions": A_test,
        "magnitudes": M_test,
    }
    torch.save(outputs, Path(args.output_dir) / f"outputs_linprobe_{args.cls_features}:{args.cca_bias}.pth")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



def collect_features(
        model: models_vit.VisionTransformer, loader: torch.utils.data.DataLoader,
        device, shuffle_subsets: int, return_features: str, tqdm_desc: str = None,
):
    model.eval()
    with torch.no_grad():
        features = []
        labels = []
        attns_list = []
        magn_list = []


        for i, (data, target) in enumerate(tqdm(loader, desc=tqdm_desc)):
            z, attns, magnitudes = model.forward_features(data.to(device), shuffle_subsets=shuffle_subsets, return_features=return_features)

            cls_cls_attns = attns[0, :, :, :, :1]
            pos_self_attns = attns[0, :, :, :, 1:].mean(dim=3, keepdim=True)

            cls_cls_attns_adj = attns[1, :, :, :, :1]
            pos_self_attns_adj = attns[1, :, :, :, 1:].mean(dim=3, keepdim=True)


            cls_pos_attns = attns[2, :, :, :, 1:].mean(dim=3, keepdim=True)
            pos_cls_attns = attns[3, :, :, :, 1:].mean(dim=3, keepdim=True)

            cls_pos_entropy = attns[4, :, :, :, :1]
            pos_pos_entropy = attns[4, :, :, :, 1:].mean(dim=3, keepdim=True)

            attn_stats = torch.cat([cls_cls_attns, pos_self_attns, cls_cls_attns_adj, pos_self_attns_adj, cls_pos_attns, pos_cls_attns, cls_pos_entropy, pos_pos_entropy], dim=3)

            magn_residual = magnitudes[0]
            magn_attended = magnitudes[1]
            magn_stats = magn_attended / (magn_residual + 1e-6)
            cls_magn_stats = magn_stats[:, :, :1]
            pos_magn_stats = magn_stats[:, :, 1:].mean(dim=2, keepdim=True)

            magn_stats = torch.cat([cls_magn_stats, pos_magn_stats], dim=2)


            features.append(z.detach().cpu())
            labels.append(target.detach().short().cpu())
            attns_list.append(attn_stats.detach().cpu())
            magn_list.append(magn_stats.detach().cpu())


            # debugging only!
            # # TODO
            # if i > 2:
            #     break

    features = torch.cat(features, dim=0)
    labels = torch.cat(labels, dim=0).long()
    attns_list = torch.cat(attns_list, dim=0)
    magns_list = torch.cat(magn_list, dim=0)

    return features, labels, attns_list, magns_list

class AggHead(nn.Module):
    def __init__(self, mlp: nn.Linear, agg_method: str):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(mlp.in_features, affine=False, eps=1e-6)
        self.mlp = mlp
        self.agg_method = agg_method

    def forward(self, X):
        B, D = X.shape

        # print(X.shape)
        if self.agg_method == "rep":
            # X = X.mean(dim = 1)
            Z = self.mlp(self.bn(X))
            return Z

        elif self.agg_method == "log":
            # X = X.reshape(B*S, D)
            # assert False, X.shape
            Z = self.mlp(self.bn(X))
            # assert False, Z.shape
            # Z = Z.reshape(B, S, Z.shape[-1])
            # Z = Z.mean(dim=1)
            # assert False, Z.shape
            return Z

        if self.agg_method == "t1":
            # X = X[:, 0]  # take the first of the shuffled representations
            Z = self.mlp(self.bn(X))
            return Z

        raise NotImplementedError(self.agg_method)



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
