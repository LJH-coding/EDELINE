from .atari_env import make_atari_env
from .minigrid_env import make_minigrid_env

def make_env(num_envs, device, cfg):
    if cfg.id.startswith("MiniGrid"):
        return make_minigrid_env(num_envs=num_envs, device=device, **cfg)
    else:
        return make_atari_env(num_envs=num_envs, device=device, **cfg)
