from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torcheval.metrics.functional import multiclass_confusion_matrix

from .blocks import Conv3x3, SmallResBlock
from data import Batch
from utils import ComputeLossOutput, init_lstm

from mambapy.mamba import Mamba, MambaConfig, RMSNorm

@dataclass
class RecurrentEmbeddingConfig:
    img_channels: int
    img_size: int
    cond_channels: int
    channels: List[int]
    down: List[int]
    mamba_config: Dict
    num_actions: Optional[int] = None

class RecurrentEmbeddingModule(nn.Module):
    def __init__(self, cfg: RecurrentEmbeddingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = RecurrentEmbeddingEncoder(cfg)
        self.act_emb = nn.Embedding(cfg.num_actions, cfg.cond_channels)
        input_dim = cfg.channels[-1] * (cfg.img_size // 2 ** (sum(cfg.down))) ** 2
        self.mamba_config = MambaConfig(**cfg.mamba_config)

        self.embedding_proj = nn.Sequential(
            nn.Linear(input_dim + cfg.cond_channels, cfg.mamba_config.d_model),
            nn.SiLU(),
        )

        self.mamba = Mamba(self.mamba_config)

        self.norm_f = RMSNorm(self.mamba_config.d_model, 1e-5, False)

    def forward(self, obs: Tensor, act: Tensor) -> Tensor:
        # print(f"RecurrentEmbeddingModule forward obs.shape = {obs.shape}, act.shape{act.shape}")
        # obs (batch_size, seq_len, 3, img_size, img_size)
        # act (batch_size, seq_len, 1)
        assert obs.ndim == 5
        b, t, c, h, w = obs.shape
        obs, act = obs.reshape(b * t, c, h, w), act.reshape(b * t)
        x = self.encoder(obs)
        act_emb = self.act_emb(act)

        x = x.reshape(b * t, -1)
        x = torch.cat((x, act_emb), dim=-1)

        x = x.reshape(b, t, -1)
        x = self.mamba(self.embedding_proj(x))
        x = self.norm_f(x)
        return x

    def step(self, obs: Tensor, act: Tensor, caches: Optional[List[Tuple[Optional[Tensor], Tensor]]] = None) -> Tuple[Tensor, List[Tuple[Optional[Tensor], Tensor]]]:
        if caches == None:
            caches = self.initialize_caches(obs.size(0))
        # obs (batch_size, 3, img_size, img_size)
        # act (batch_size, 1)
        assert obs.ndim == 4
        b, c, h, w = obs.shape
        x = self.encoder(obs)
        act_emb = self.act_emb(act)

        x = x.reshape(b, -1)
        x = torch.cat((x, act_emb), dim=-1)

        x, caches = self.mamba.step(self.embedding_proj(x), caches)
        x = self.norm_f(x)
        return x, caches

    def process_sequence(self, obs: Tensor, act: Tensor) -> Tuple[Tensor, List[Tuple[Optional[Tensor], Tensor]]]:
        # obs: (batch_size, seq_len, c, h, w)
        # act: (batch_size, seq_len)
        assert obs.ndim == 5 and act.ndim == 2
        b, t, c, h, w = obs.shape
        x = []
        caches = self.initialize_caches(b)
        for i in range(t):
            xi, caches = self.step(obs[:, i], act[:, i], caches)
            x.append(xi)
        x = torch.stack(x, dim=1)
        return x, caches

    def initialize_caches(self, batch_size):
        caches = []
        for layer in self.mamba.layers:
            mamba_block = layer.mixer  # Access the MambaBlock directly
            
            # Initialize h as None
            h = None
            
            # Initialize inputs for the conv1d
            # Shape: (batch_size, d_inner, d_conv-1)
            inputs = torch.zeros(
                batch_size, 
                mamba_block.config.d_inner,  # Use d_inner instead of d_model
                mamba_block.config.d_conv - 1,
                device=next(self.mamba.parameters()).device
            )
            
            # Append the cache tuple for this layer
            caches.append((h, inputs))
        
        return caches

class RecurrentEmbeddingEncoder(nn.Module):
    def __init__(self, cfg: RecurrentEmbeddingConfig) -> None:
        super().__init__()
        assert len(cfg.channels) == len(cfg.down)
        encoder_layers = [Conv3x3(cfg.img_channels, cfg.channels[0])]
        for i in range(len(cfg.channels)):
            encoder_layers.append(SmallResBlock(cfg.channels[max(0, i - 1)], cfg.channels[i]))
            if cfg.down[i]:
                encoder_layers.append(nn.MaxPool2d(2))
        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)
