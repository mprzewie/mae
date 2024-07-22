import numpy as np
import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

from util.misc import is_dist_avail_and_initialized


def uniformity_loss(features):
    # gather across devices
    features = torch.cat(GatherLayer.apply(features), dim=0)
    # calculate loss
    features = torch.nn.functional.normalize(features)
    sim = features @ features.T
    loss = sim.pow(2).mean()
    return loss

class ClsPosLoss(nn.Module):
    def __init__(
            self,
            loss_type: str, out_dim: int,
            norm_targets: bool,
            *,
            number_of_epochs,
            warmup_teacher_temp_epochs,
            warmup_teacher_temp=0.04,
            teacher_temp=0.04,
            center_momentum=0.9
    ):
        super().__init__()
        self.loss_type = loss_type
        self.norm_targets = norm_targets


        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(number_of_epochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.center_momentum = center_momentum

    def forward(self, t_latent, p_latent, epoch: int):
        t_latent = t_latent.detach()
        assert t_latent.shape == p_latent.shape

        if self.norm_targets:
            mean = t_latent.mean(dim=-1, keepdim=True)
            var = t_latent.var(dim=-1, keepdim=True)
            t_latent = (t_latent - mean) / (var + 1.e-6) ** .5

        if self.loss_type == "mse":
            loss_latent = (p_latent - t_latent).pow(2).mean()
        elif self.loss_type == "cos":
            loss_latent = - torch.nn.functional.cosine_similarity(p_latent, t_latent, dim=-1).mean()
        elif self.loss_type == "dino":
            B, T, D = t_latent.shape
            t_latent = t_latent.reshape(B * T, D)
            p_latent = p_latent.reshape(B * T, D)
            temp = self.teacher_temp_schedule[epoch]
            t = F.softmax((t_latent - self.center) / temp, dim=-1)
            loss_latent = (-t * F.log_softmax(p_latent, dim=-1)).sum(dim=-1).mean()
            self.update_center(t_latent)

        else:
            raise NotImplementedError(self.loss_type)

        return loss_latent

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        ws = 1
        if is_dist_avail_and_initialized():
            dist.all_reduce(batch_center)
            ws = dist.get_world_size()

        batch_center = batch_center / (len(teacher_output) * ws)
        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)



def entropy_loss(cls_features, epsilon: float=1e-6):
    sft = torch.nn.functional.softmax(cls_features, dim=1) + epsilon
    individual_entropy = - (sft * sft.log()).sum(dim=1).mean()
    # minimize individual entropy (each sample should activate one output strongly)
    mca = sft.mean(dim=0)
    batch_entropy = (mca * mca.log()).sum()
    # *maximize* batch entropy (all outputs should be activated ~equally by examples in the batch)
    return individual_entropy, batch_entropy

class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input.contiguous())
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out