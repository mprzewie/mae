"""
Adapted from https://github.com/gmum/Kernel_SA-AbMILP
"""

import torch.nn as nn
import torch.nn.functional as F

from models_vit import Attention


class ABMILPHead(nn.Module):
    def __init__(self, dim: int, self_attn: bool = False, activation: str= "tanh"):
        super().__init__()

        self.self_attn = Attention(dim) if self_attn else nn.Identity()

        self.ATTENTION_BRANCHES = 1

        self.attention = nn.Sequential(
            nn.Linear(dim, dim), # matrix V
            (nn.Tanh() if activation == "tanh" else nn.ReLU()),
            nn.Linear(dim, self.ATTENTION_BRANCHES)
        )

    def forward(self, x):
        x_attn = self.self_attn(x)
        if isinstance(x_attn, tuple):
            x_attn = x_attn[0]

        attn_w = self.attention(x_attn)

        attn_sft = F.softmax(attn_w, dim=1)

        ret = (x_attn * attn_sft).sum(dim=1)
        return ret



