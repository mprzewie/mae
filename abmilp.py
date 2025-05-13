"""
Adapted from https://github.com/gmum/Kernel_SA-AbMILP
"""
from typing import Optional, List

import torch.nn
import torch.nn as nn
import torch.nn.functional as F

from models_vit import Attention
from util.pos_embed import get_2d_sincos_pos_embed


class ABMILPHead(nn.Module):
    def __init__(
            self,
            dim: int,
            self_attention_apply_to: str = "none",
            activation: str= "tanh",
            depth: int = 2,
            cond: str="none",
            content: str = "all",
            num_patches: Optional[int] = None,
            num_heads: Optional[int] = 8,
            attention_branches: int = 1,
        ):
        super().__init__()

        self.cond = cond
        self.self_attention_apply_to = self_attention_apply_to
        self.content = content
        self.pos_embed = torch.nn.Parameter(
            torch.from_numpy(
                get_2d_sincos_pos_embed(dim, int(num_patches ** .5), cls_token=(content!="patch"))
            ).float().unsqueeze(0),
            requires_grad=False
        )

        self.self_attn = Attention(dim, num_heads=num_heads) if self.self_attention_apply_to != "none" else nn.Identity()


        # self.ATTENTION_BRANCHES = 1

        attn_pred_layers = []
        for i in range(depth-1):
            attn_pred_layers.extend([
                nn.Linear(dim, dim),
                (nn.Tanh() if activation == "tanh" else nn.ReLU() if activation == "relu" else nn.GELU()),
            ])

        attn_pred_layers.append(nn.Linear(dim, attention_branches))
        self.attention_predictor = nn.Sequential(*attn_pred_layers)

    def forward_with_attn_map(self, x):
        if self.content == "patch":
            x = x[:, 1:] # keep patch tokens only

        x_attn = self.self_attn(x)
        if isinstance(x_attn, tuple):
            x_attn = x_attn[0]

        predictor_input = x_attn if self.self_attention_apply_to in ["map", "both"] else x
        if self.cond == "pe":
            predictor_input = predictor_input + self.pos_embed

        attn_map = self.attention_predictor(predictor_input)
        attn_map = F.softmax(attn_map, dim=1)

        x_out = x_attn if self.self_attention_apply_to in ["both"] else x
        # out = (x_out * attn_map).sum(dim=1)

        x_perm = x_out.permute(0, 2, 1)

        x_attn = (x_perm @ attn_map)

        if x_attn.shape[2] == 1:
            x_attn = x_attn.squeeze(2)

        return x_attn, attn_map

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.forward_with_attn_map(x)
        return out




class AbMILPCelebAHead(nn.Module):
    def __init__(self, in_features: int, n_tasks: int) -> None:
        super().__init__()
        # solves each binary classification task separately
        self.W = nn.Parameter(torch.randn(n_tasks, in_features, 1))
        self.b = nn.Parameter(torch.randn(n_tasks))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xW = (x.permute(2, 0, 1) @ self.W)
        xW = xW.squeeze().T
        x_out = xW + self.b
        return x_out
