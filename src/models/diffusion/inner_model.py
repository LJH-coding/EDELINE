from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import Conv3x3, FourierFeatures, GroupNorm, UNet, ResBlocks, Downsample
from utils import init_lstm

@dataclass
class InnerModelConfig:
    img_channels: int
    num_steps_conditioning: int
    mamba_dim: int
    cond_channels: int
    depths: List[int]
    channels: List[int]
    attn_depths: List[bool]
    num_actions: Optional[int] = None

class InnerModel(nn.Module):
    def __init__(self, cfg: InnerModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.noise_emb = FourierFeatures(cfg.cond_channels)
        self.embedding_proj = nn.Linear(cfg.mamba_dim, cfg.cond_channels)

        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )
        self.conv_in = Conv3x3((cfg.num_steps_conditioning + 1) * cfg.img_channels, cfg.channels[0])

        self.unet = UNet(cfg.cond_channels, cfg.depths, cfg.channels, cfg.attn_depths)

        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = Conv3x3(cfg.channels[0], cfg.img_channels)
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, noisy_next_obs: Tensor, c_noise: Tensor, obs: Tensor, emb: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        # print(f"Inner_Model forward, noisey_next_obs.shape = {noisy_next_obs.shape}, obs.shape = {obs.shape}, emb.shape = {emb.shape}")
        cond = self.cond_proj(self.noise_emb(c_noise) + self.embedding_proj(emb))
        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1))
        x, _, _ = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x
