"""Adapted from https://github.com/JiePKU/understand-ssl-part-aware/blob/master/retrieval-classification/patch_lincls.py"""
from functools import partial

import torch
from timm.layers import trunc_normal_
from torch import nn
import torch.nn.functional as F
'''
Attention with bool_masked_pos argument.
Support cross-attention.
'''


class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)

        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            # self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.window_size = None
        self.relative_position_bias_table = None
        self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, bool_masked_pos=None, rel_pos_bias=None, k=None, v=None):
        B, N, C = x.shape

        if k is None:
            k = x
            v = x
            N_k = N
            N_v = N
        else:
            N_k = k.shape[1]
            N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            v_bias = self.v_bias

        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)  # (B, N_q, dim)
        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)  # (B, N_k, dim)
        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)

        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, num_heads, N_q, dim)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, num_heads, N_k, dim)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, num_heads, N_v, dim)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N, N)

        # NOTE: relative position bias
        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class DecoderBlockSimple(nn.Module):

    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None):
        super().__init__()

        # NOTE: cross attention
        self.norm1_q_cross = norm_layer(dim)
        self.norm1_k_cross = norm_layer(dim)
        self.norm1_v_cross = norm_layer(dim)
        self.norm2_cross = norm_layer(dim)
        self.cross_attn =  CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, window_size=window_size, attn_head_dim=attn_head_dim)

    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos, rel_pos_bias=None):
        x_q = self.norm1_q_cross(x_q + pos_q)
        x_k = self.norm1_k_cross(x_kv + pos_k)
        x_v = self.norm1_v_cross(x_kv)
        x = self.cross_attn(x_q, bool_masked_pos, rel_pos_bias=rel_pos_bias, k=x_k, v=x_v)

        return x

class AttentiveHead(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 12):
        super(AttentiveHead, self).__init__()
        self.cross_attn = DecoderBlockSimple(
            embed_dim, num_heads=num_heads, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.query_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cross_attn.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    def forward(self, x):
        query_tokens = self.query_token.expand(x.size(0), -1, -1)
        query_tokens = self.cross_attn(query_tokens, x, 0, 0, bool_masked_pos=None, rel_pos_bias=None)
        return query_tokens[:, 0]
