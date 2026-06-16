import torch
import torch.nn.functional as F
import torch.nn as nn
from vimo.backbone_vimo import Qwen3VLVisionBackbone

class DistillLoss(torch.nn.Module):
    def __init__(self, config, stage=1, normalize_feats: bool = True, norm_eps: float = 1e-6):
        super().__init__()
        self.stage = stage
        self.normalize_feats = normalize_feats
        self.norm_eps = norm_eps

        if self.stage == 2:
            self.backbone = Qwen3VLVisionBackbone(config)
            self.backbone.freeze()
            self.backbone.eval()

        self.cos = nn.CosineSimilarity(dim=2, eps=1e-6)

    def _maybe_norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, C] (or generally normalize along last dim)
        if not self.normalize_feats:
            return x
        return F.normalize(x, p=2, dim=-1, eps=self.norm_eps)

    def forward(self, hidden_states, grid_thw, latent, dinov2):
        loss = 0.0

        if self.stage == 2:
            self.eval()
            with torch.no_grad():
                hidden_states, process_hidden_states = self.backbone(hidden_states, grid_thw)
                gen_bs = grid_thw.shape[0]
                gen_hidden_states = hidden_states.view(gen_bs, -1, 1024)

            target_feat = gen_hidden_states

            # optional normalization
            latent_n = self._maybe_norm(latent)
            target_n = self._maybe_norm(target_feat)
            dinov2_n = self._maybe_norm(dinov2)

            distill_loss = F.mse_loss(latent_n, target_n, reduction="mean")
            loss += distill_loss

            sim_loss = 1 - self.cos(dinov2_n, target_n)

        else:
            sim_loss = 1 - self.cos(dinov2, latent)

        loss += sim_loss.mean()
        return loss