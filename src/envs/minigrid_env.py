from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import gymnasium
from gymnasium.vector import AsyncVectorEnv
from minigrid.wrappers import RGBImgPartialObsWrapper
import numpy as np
import torch
from torch import Tensor
import cv2

def make_minigrid_env(
    id: str,
    num_envs: int,
    device: torch.device,
    size: int,
    max_episode_steps: Optional[int],
) -> TorchEnv:
    def env_fn():
        env = gymnasium.make(
            id,
            render_mode="rgb_array",
        )
        env = RGBImgPartialObsWrapper(env, tile_size=16)
        env = MinigridEnv(env, size=size)
        return env

    env = AsyncVectorEnv([env_fn for _ in range(num_envs)])

    env = TorchEnv(env, device)

    return env

class MinigridEnv(gymnasium.Wrapper):
    def __init__(self, env, size: int):
        super().__init__(env)
        self.size = size
        self.observation_space = gymnasium.spaces.Box(
            low=0, high=255, shape=(size, size, 3), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self._process_obs(obs), reward, done, truncated, info

    def _process_obs(self, obs):
        obs = obs['image']
        obs = cv2.resize(
            obs,
            (self.size, self.size),
            interpolation=cv2.INTER_AREA
        )
        
        return obs

class TorchEnv(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, device: torch.device) -> None:
        super().__init__(env)
        self.device = device
        self.num_envs = env.observation_space.shape[0]
        self.num_actions = env.unwrapped.single_action_space.n
        b, h, w, c = env.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(b, c, h, w))

    def reset(self, *args, **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        obs, info = self.env.reset(*args, **kwargs)
        return self._to_tensor(obs), info

    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        obs, rew, end, trunc, info = self.env.step(actions.cpu().numpy())
        dead = np.logical_or(end, trunc)
        if dead.any():
            info["final_observation"] = self._to_tensor(np.stack(info["final_observation"][dead]))
        obs, rew, end, trunc = (self._to_tensor(x) for x in (obs, rew, end, trunc))
        return obs, rew, end, trunc, info

    def _to_tensor(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            return torch.tensor(x, device=self.device).div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()
        elif x.dtype is np.dtype("bool"):
            return torch.tensor(x, dtype=torch.uint8, device=self.device)
        else:
            return torch.tensor(x, dtype=torch.float32, device=self.device)
