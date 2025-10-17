from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_confusion_matrix

from .blocks import Conv3x3, Downsample, ResBlocks
from data import Batch
from utils import ComputeLossOutput, init_lstm

@dataclass
class RewEndModelConfig:
    mamba_dim: int
    num_actions: Optional[int] = None

class RewEndModel(nn.Module):
    def __init__(self, cfg: RewEndModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.head_rew = nn.Sequential(
            nn.Linear(cfg.mamba_dim, cfg.mamba_dim),
            nn.SiLU(),
            nn.Linear(cfg.mamba_dim, 3, bias=False)
        )
        self.head_end = nn.Sequential(
            nn.Linear(cfg.mamba_dim, cfg.mamba_dim),
            nn.SiLU(),
            nn.Linear(cfg.mamba_dim, 2, bias=False)
        )

    def forward(
        self,
        emb: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # print(f"RewEndModel forward, next_obs.shape = {next_obs.shape}, emb.shape = {emb.shape}")
        logits_rew = self.head_rew(emb)
        logits_end = self.head_end(emb)
        return logits_rew, logits_end

    def compute_loss(self, batch: Batch, batch_emb):
        obs = batch.obs
        act = batch.act
        rew = batch.rew
        end = batch.end
        mask = batch.mask_padding

        b, t, c, h, w = obs.shape
        
        emb = batch_emb.reshape(b * t, -1)

        logits_rew, logits_end = self(emb)
        logits_rew, logits_end = logits_rew.reshape(b, t, -1), logits_end.reshape(b, t, -1)
        logits_rew = logits_rew[mask]
        logits_end = logits_end[mask]
        target_rew = rew[mask].sign().long().add(1)  # clipped to {-1, 0, 1}
        target_end = end[mask]

        loss_rew = F.cross_entropy(logits_rew, target_rew)
        loss_end = F.cross_entropy(logits_end, target_end)

        metrics = {
            "confusion_matrix": {
                "rew": multiclass_confusion_matrix(logits_rew, target_rew, num_classes=3),
                "end": multiclass_confusion_matrix(logits_end, target_end, num_classes=2),
            },
        }
        return loss_rew, loss_end, metrics