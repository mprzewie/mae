# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
import math
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
from typing import Optional, Final, Type

import numpy as np
import torch
import torch.nn as nn

import timm.models.vision_transformer
import torch.nn.functional as F

from time import time
from pprint import pprint

class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f'{dim=} should be divisible by {num_heads=}'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = False  # use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # self.register_buffer("cls_bias", torch.zeros(1))
        # assert False, self.cls_bias

        self.cls_bias = None #torch.zeros(num_heads).cuda() if torch.cuda.is_available() else torch.zeros(num_heads)# TODO maybe a register

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = time()
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            assert False
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # s = time()
            if self.cls_bias is not None:
                cb = self.cls_bias #.to(attn.device)
                # t1 = time()

                attn[:, :, 0, 0] += cb
                # t2 = time()
                # attn[:, :, 0, 0] = attn[:, :, 0, 0].clamp(0 , 1)
                attn = attn.clamp(0, 1)
                # t3 = time()

                target_non_cc_weight = 1 - attn[:, :, 0, 0]
                actual_non_cc_weight = attn[:, :, 0, 1:].sum(dim=2)
                epsilon = 1e-6
                mp = target_non_cc_weight / (actual_non_cc_weight + epsilon)

                attn[:, :, 0, 1:] *= mp.unsqueeze(2)
                attn = attn.clamp(0, 1)

            x = attn @ v
            x = x.transpose(1, 2).reshape(B, N, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            # sattn = time()
            # print("attn total", sattn - s0)
            return x, attn




class Block(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = timm.models.vision_transformer.Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = nn.Identity()
        self.drop_path1 = nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = nn.Identity()
        self.drop_path2 = nn.Identity()

    def forward(self, x: torch.Tensor, return_attention=False) -> torch.Tensor:
        y, attention = self.attn(self.norm1(x))

        x_norm = torch.linalg.vector_norm(x, dim=2)
        y_norm = torch.linalg.vector_norm(y, dim=2)

        magnitudes = torch.stack((x_norm, y_norm), dim=0)


        x = x + self.drop_path1(self.ls1(y))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        if return_attention:
            return x, attention, magnitudes

        return x

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(
            self,
            n_last_layers: int = 1,
            block_reshuffling: bool = False,
            global_pool=False,
            block_fn: Type[nn.Module]=Block,
            **kwargs
    ):
        super(VisionTransformer, self).__init__(block_fn=block_fn, **kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

        self.n_last_layers = n_last_layers
        self.head = nn.Linear(
            self.embed_dim * n_last_layers, self.num_classes
        )
        self.block_reshuffling = block_reshuffling
        assert not self.block_reshuffling

    def forward_features(self, x, shuffle_subsets: int = 1, return_features: str = "cls", return_final_attn: bool = False):
        # assert shuffle_subsets == 1, shuffle_subsets
        orig_x = x
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x_cls = x[:, :1]
        x_pos = x[:, 1:]

        assert x_pos.shape[1] % shuffle_subsets == 0, f"{x_pos.shape[1]=} not divisible by {shuffle_subsets=}"
        x_cls = x_cls.unsqueeze(1).repeat(1, shuffle_subsets, 1, 1)
        B, L, D = x_pos.shape

        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]
        if shuffle_subsets == 1:
            noise, _ = torch.sort(noise, dim=1)
            # if we dont shuffle subsets, undo shuffling

        ids_shuffle = torch.argsort(noise, dim=1)
        x_pos_shuffled = torch.gather(x_pos, dim=1,index=ids_shuffle.unsqueeze(-1).repeat(1, 1, D))
        x_pos_shuffled = x_pos_shuffled.reshape(B, shuffle_subsets, L // shuffle_subsets, D)

        x = torch.cat([x_cls, x_pos_shuffled], dim=2).reshape(
            B*shuffle_subsets, (L//shuffle_subsets)+1, D
        )

        attentions = []
        magnitudes = []
        for blk in self.blocks:
            x, attn, magn = blk.forward(x, return_attention=True)

            _, _, T, T = attn.shape
            attn_range = torch.arange(T)
            attn_diag = attn[:, :, attn_range, attn_range] # attention of tokens w.r.t. themselves
            cls_all_attn = attn[:, :, 0, ]  # attention of cls token to all tokens
            all_cls_attn = attn[:, :, :, 0] # attention of all tokens to cls token

            attn_wo_cls = attn[:, :, :, 1:]
            attn_wo_cls_denom = attn_wo_cls.sum(dim=3, keepdim=True)

            attn_wo_cls = attn_wo_cls / (attn_wo_cls_denom + 1e-6)

            all_pos_attn_entropy = -(attn_wo_cls * (attn_wo_cls + 1e-6).log()).sum(dim=3)

            attn_adj_for_cls = attn / (attn_wo_cls_denom + 1e-6)

            attn_diag_adj_for_cls = attn_adj_for_cls[:, :, attn_range, attn_range]

            attn_stats = torch.stack([attn_diag, attn_diag_adj_for_cls, cls_all_attn, all_cls_attn, all_pos_attn_entropy])

            attn_stats = attn_stats.unsqueeze(2)

            # assert False, attn_stats.shape
            attentions.append(attn_stats.detach())
            magnitudes.append(magn.unsqueeze(2).detach())

            # if self.block_reshuffling:
            #     x_n_s_cl_d = x.reshape(B, shuffle_subsets, (L // shuffle_subsets) + 1, D)
            #     x_cls = x_n_s_cl_d[:, :, :1]
            #     x_pos = x_n_s_cl_d[:, :, 1:].reshape(B, L, D)
            #     noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]
            #     ids_shuffle = torch.argsort(noise, dim=1)
            #     x_pos_shuffled = torch.gather(x_pos, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, D))
            #     x_pos_shuffled = x_pos_shuffled.reshape(B, shuffle_subsets, L // shuffle_subsets, D)
            #     x = torch.cat([x_cls, x_pos_shuffled], dim=2).reshape(
            #         B * shuffle_subsets, (L // shuffle_subsets) + 1, D
            #     )

        x_n_s_cl_d = x.reshape(B, shuffle_subsets, (L//shuffle_subsets)+1, D)

        x_cls = x_n_s_cl_d[:, :, 0]
        x_pos = x_n_s_cl_d[:, :, 1:].mean(dim=2)

        # average back the shuffled subsets
        x_cls = x_cls.mean(dim=1)
        x_pos = x_pos.mean(dim=1)

        if return_features == "cls":
            ret = x_cls
        elif return_features == "pos":
            ret = x_pos
        elif return_features == "both":
            ret = torch.concat([x_cls, x_pos], dim=2)
        elif return_features.startswith("cp"):
            assert shuffle_subsets==1
            cp = int(return_features.split("cp")[1])
            B, SS, T1, D = x_n_s_cl_d.shape
            x_n_cl_d = x_n_s_cl_d[:, 0]
            fm = x_n_cl_d[:, 1:]
            T = fm.shape[1]
            hw = np.sqrt(T)
            assert int(hw) == hw, hw
            hw = int(hw)
            c = hw // 2
            s = c - math.ceil(cp/2)
            e = c + math.floor(cp/2)
            fm = fm.reshape(B, hw, hw, D)
            fm = fm[:, s:e, s:e]
            fm = fm.mean(dim=[1,2])
            ret = fm

        elif return_features == "dino":
            assert shuffle_subsets == 1
            x_n_cl_d = x_n_s_cl_d[:, 0]
            fm = x_n_cl_d[:, 1:]

            if orig_x.device==torch.device("cuda"):
                _, _, _, d_attn = _DINO_cuda.forward_features(orig_x, return_final_attn=True)
            else:
                _, _, _, d_attn = _DINO_cpu.forward_features(orig_x,  return_final_attn=True)

            d_attn = d_attn[:, :, 0, 1:].unsqueeze(3)
            fm = fm.unsqueeze(1)

            fm_mul = fm * d_attn

            ret = fm_mul.mean(dim=[1,2])

        else:
            raise NotImplementedError(return_features)

        attentions = torch.cat(attentions, dim=2) # kind, batch, blocks, heads, tokens
        magnitudes = torch.cat(magnitudes, dim=2) # kind, batch, blocks, tokens

        if return_final_attn:
            return ret, attentions, magnitudes, attn

        return ret, attentions, magnitudes



    def forward(self, x: torch.Tensor, return_features: str = "cls") -> torch.Tensor:
        x, attn, magnitudes = self.forward_features(x, return_features=return_features, shuffle_subsets=1)
        # x = x.mean(dim=1)  # account for shuffle subsets which is essentially a no-op in this case
        x = self.head(x)
        return x





def vit_tiny_patch16(img_size=224, patch_size=16, **kwargs):
    model = VisionTransformer(
        img_size=img_size, patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small_patch16(img_size=224, patch_size=16, **kwargs):
    model = VisionTransformer(
        img_size=img_size, patch_size=patch_size, embed_dim=384, depth=12, num_heads=6,mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16(img_size=224, patch_size=16, **kwargs):
    model = VisionTransformer(
        img_size=img_size, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_large_patch16(**kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def vit_huge_patch14(**kwargs):
    model = VisionTransformer(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def _timm(name="vit_base_patch16_224.dino") -> VisionTransformer:
    from timm.models.vision_transformer import _create_vision_transformer
    checkpoint_model = _create_vision_transformer(name, pretrained=True, patch_size=16, embed_dim=768, depth=12,
                                                  num_heads=12).state_dict()
    model = vit_base_patch16()
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    from util.pos_embed import interpolate_pos_embed
    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(name, msg)
    return model

_DINO_cpu = _timm()

try:
    _DINO_cuda = _timm().cuda()
except:
    print("Can't initialize dino cuda")