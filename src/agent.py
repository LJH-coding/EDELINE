from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from envs import TorchEnv, WorldModelEnv
from models.world_model import WorldModel, WorldModelConfig
from models.actor_critic import ActorCritic, ActorCriticConfig, ActorCriticLossConfig
from models.diffusion import SigmaDistributionConfig
from utils import extract_state_dict


@dataclass
class AgentConfig:
    world_model: WorldModelConfig
    actor_critic: ActorCriticConfig
    num_actions: int

    def __post_init__(self) -> None:
        self.world_model.denoiser.inner_model.num_actions = self.num_actions
        self.world_model.rew_end_model.num_actions = self.num_actions
        self.actor_critic.num_actions = self.num_actions
        self.world_model.recurrent_embedding_module.num_actions = self.num_actions


class Agent(nn.Module):
    def __init__(self, cfg: AgentConfig) -> None:
        super().__init__()
        self.world_model = WorldModel(cfg.world_model)
        self.actor_critic = ActorCritic(cfg.actor_critic)

    @property
    def device(self):
        return self.world_model.denoiser.device

    def setup_training(
        self,
        sigma_distribution_cfg: SigmaDistributionConfig,
        actor_critic_loss_cfg: ActorCriticLossConfig,
        rl_env: Union[TorchEnv, WorldModelEnv],
    ) -> None:
        self.world_model.denoiser.setup_training(sigma_distribution_cfg)
        self.actor_critic.setup_training(rl_env, actor_critic_loss_cfg)

    def load(
        self,
        path_to_ckpt: Path,
        load_world_model: bool = True,
        load_actor_critic: bool = True,
    ) -> None:
        sd = torch.load(Path(path_to_ckpt), map_location=self.device)
        sd = {k: extract_state_dict(sd, k) for k in ("world_model", "actor_critic")}
        if load_world_model:
            self.world_model.load_state_dict(sd["world_model"])
        if load_actor_critic:
            self.actor_critic.load_state_dict(sd["actor_critic"])
