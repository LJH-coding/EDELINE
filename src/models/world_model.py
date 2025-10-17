import torch.nn as nn
import torch
from dataclasses import dataclass
from models.diffusion import Denoiser, DenoiserConfig
from models.recurrent_embedding_module import RecurrentEmbeddingModule, RecurrentEmbeddingConfig
from models.rew_end_model import RewEndModel, RewEndModelConfig

@dataclass
class WorldModelConfig:
    harmony: bool
    denoiser: DenoiserConfig
    recurrent_embedding_module: RecurrentEmbeddingConfig
    rew_end_model: RewEndModelConfig

class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.recurrent_emb = RecurrentEmbeddingModule(cfg.recurrent_embedding_module)
        self.denoiser = Denoiser(cfg.denoiser)
        self.rew_end_model = RewEndModel(cfg.rew_end_model)
        if cfg.harmony:
            self.harmony_s1 = nn.Parameter(-torch.log(torch.tensor(1.0)))  # rew_loss
            self.harmony_s2 = nn.Parameter(-torch.log(torch.tensor(1.0)))  # denoiser_loss

    def compute_loss(self, batch):
        batch_emb = self.recurrent_emb(batch.obs, batch.act)
        rew_loss, end_loss, rew_end_model_metrics = self.rew_end_model.compute_loss(batch, batch_emb)
        denoiser_loss = self.denoiser.compute_loss(batch, batch_emb)

        if self.cfg.harmony:
            rew_loss = rew_loss / torch.exp(self.harmony_s1)
            denoiser_loss = denoiser_loss / torch.exp(self.harmony_s2)
            loss = rew_loss + denoiser_loss + end_loss
            harmony_reg = (torch.log(torch.exp(self.harmony_s1) + 1) +
                           torch.log(torch.exp(self.harmony_s2) + 1))
            loss += harmony_reg
            metrics = {
                "harmony_s1": self.harmony_s1.item(),
                "coeff1": (1 / torch.exp(self.harmony_s1)).item(),
                "sigma1": torch.exp(self.harmony_s1 * 0.5).item(),
                "harmony_base1": torch.log(torch.exp(self.harmony_s1) + 1).item(),
                "harmony_s2": self.harmony_s2.item(),
                "coeff2": (1 / torch.exp(self.harmony_s2)).item(),
                "sigma2": torch.exp(self.harmony_s2 * 0.5).item(),
                "harmony_base2": torch.log(torch.exp(self.harmony_s2) + 1).item()
            }
        else:
            loss = rew_loss + denoiser_loss + end_loss
            metrics = {}

        metrics.update({
            "loss_world_model": loss.item(),
            "loss_rew": rew_loss.detach(),
            "loss_end": end_loss.detach(),
            "loss_denoising": denoiser_loss.detach(),
            **rew_end_model_metrics
        })


        return loss, metrics
    