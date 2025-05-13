"Adapted from https://github.com/facebookresearch/capi/blob/main/eval_classification.py#L237"
import torch
from torch import nn


class AllClassifiers(nn.Module):
    def __init__(self, classifiers: dict[str, nn.Module]):
        super().__init__()
        self.classifiers = nn.ModuleDict(classifiers)

    def forward(self, backbone_out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: self.classifiers[key](backbone_out[("abmilp" if key.startswith("abmilp") else key)])
            for key in self.classifiers
        }