import torch
from torch import nn

from models_vit import VisionTransformer


class LatentMixer(nn.Module):
    def __init__(
            self,
            encoder: VisionTransformer,
            discriminator: VisionTransformer,
            mixing_rate: float = 0.5,
            mixing_strategy: str = "seq",
    ):
        super(LatentMixer, self).__init__()
        self.encoder = encoder
        self.discriminator = discriminator

        self.proj = nn.Linear(encoder.embed_dim, discriminator.embed_dim)
        self.bin_classifier = nn.Linear(discriminator.embed_dim, 1) # for loss
        self.lin_classifier = nn.Linear(encoder.embed_dim, 1000) # for evaluation purposes

        self.mixing_strategy = mixing_strategy
        self.mixing_rate = mixing_rate






    def forward(self, X):
        tokens, _, _ = self.encoder.forward_features(X, return_features="raw")
        encoder_tokens = tokens

        cls_tokens = tokens[:, :1]
        patch_tokens = tokens[:, 1:]
        B, N, E = patch_tokens.shape
        X_ids = torch.arange(B).reshape(B, 1, 1).repeat(1, N, 1).to(X.device)
        X_ids_GT = X_ids.clone().squeeze()

        noise = torch.rand(B, N, device=patch_tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove

        patch_tokens_shuffled = torch.gather(
            patch_tokens, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, E)
        )

        N_mix = int(N * self.mixing_rate)
        N_keep = N - N_mix

        patch_tokens_keep = patch_tokens_shuffled[:, :N_keep]
        X_ids_keep = X_ids[:, :N_keep]

        patch_tokens_mix = patch_tokens_shuffled[:, N_keep:]
        X_ids_mix = X_ids[:, N_keep:]

        if self.mixing_strategy == "seq":
            seq_ids = torch.randperm(B).to(X.device).reshape(B, 1).repeat(1, N_mix)
            # print(seq_ids)
            X_ids_mix = torch.gather(
                X_ids_mix, dim=0, index=seq_ids.unsqueeze(-1).repeat(1, 1, 1)
            )
            # print(X_ids_mix.squeeze())
            patch_tokens_mix = torch.gather(
                patch_tokens_mix, dim=0, index=seq_ids.unsqueeze(-1).repeat(1, 1, E)
            )
            # assert False, (patch_tokens_keep.shape, patch_tokens_mix.shape)

        else:
            raise NotImplementedError(self.mixing_strategy)

        patch_tokens = torch.cat([patch_tokens_keep, patch_tokens_mix], dim=1)
        X_ids = torch.cat([X_ids_keep, X_ids_mix], dim=1).squeeze()
        # X_ids_true =
        # print(X_ids_GT)
        # print(X_ids)
        targets = (X_ids_GT == X_ids).float()
        # print(targets)
        # assert False, X_ids.squeeze()
        tokens = torch.cat([cls_tokens, patch_tokens], dim=1)

        disc_tokens = self.proj(tokens)


        for db in self.discriminator.blocks:
            disc_tokens = db.forward(disc_tokens)

        disc_patch_tokens = disc_tokens[:, 1:]
        disc_cls = self.bin_classifier(disc_patch_tokens).squeeze()

        loss = torch.nn.functional.binary_cross_entropy_with_logits(disc_cls, targets)
        acc = ((disc_cls > 0).float() == targets).float().mean()

        lin_cls_output = self.lin_classifier(encoder_tokens[:, 0].detach())

        return loss, (acc, encoder_tokens, lin_cls_output)




        # assert False, (tokens.shape, disc_tokens.shape, disc_cls.shape)