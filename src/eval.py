import torch
from agent import Agent
from envs import make_env
from pathlib import Path
from omegaconf import OmegaConf
from utils import get_path_agent_ckpt, prompt_atari_game
from hydra import compose, initialize
from hydra.utils import instantiate
import time

def evaluate_agent(agent: Agent, env, num_episodes: int = 10):
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        hx_cx = None

        while not done:
            with torch.no_grad():
                logits_act, value, hx_cx = agent.actor_critic(obs, hx_cx)
                dst = torch.distributions.Categorical(logits=logits_act)
                action = dst.sample()

            obs, reward, done, trunc, info = env.step(action)

            total_reward += reward.item()

            if done or trunc:
                print(f"Episode {episode + 1}, return: {total_reward}")
                break

def main():
    with initialize(version_base="1.3", config_path="../config"):
        cfg = compose(config_name="trainer")

    # Set up evaluation environment and agent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(num_envs=1, device=device, cfg=cfg.env.test)

    path_ckpt = get_path_agent_ckpt("checkpoints", epoch=-1)
    agent = Agent(instantiate(cfg.agent, num_actions=env.num_actions)).to(device).eval()
    agent.load(path_ckpt)

    # Evaluate the agent in the environment
    evaluate_agent(agent, env, num_episodes=20)

    env.close()

if __name__ == "__main__":
    main()
