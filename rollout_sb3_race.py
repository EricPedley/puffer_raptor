#!/usr/bin/env python3
"""Render rollouts from an SB3 checkpoint on the race env."""

import argparse
import torch
import numpy as np
from stable_baselines3 import PPO

from race_env import QuadcopterRaceEnv
from train_race import easy_pos, easy_yaw, easy_start, hard_pos, hard_yaw, hard_start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str, help="Path to SB3 .zip checkpoint")
    parser.add_argument("--track", type=str, default="easy", choices=["easy", "hard"])
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--config-path", type=str, default="meteor75_parameters.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.track == "easy":
        gate_pos, gate_yaw, start_pos = easy_pos, easy_yaw, easy_start
    else:
        gate_pos, gate_yaw, start_pos = hard_pos, hard_yaw, hard_start

    env = QuadcopterRaceEnv(
        gate_positions=gate_pos,
        gate_yaws=gate_yaw,
        start_position=start_pos,
        num_envs=args.num_envs,
        config_path=args.config_path,
        device=args.device,
        render_mode="human",
        max_episode_length=int(2e10)
    )

    print(f"Loading SB3 checkpoint: {args.checkpoint}")
    model = PPO.load(args.checkpoint, device=args.device)
    model.policy.eval()

    episode_count = 0
    total_reward = 0.0

    obs, _ = env.reset()
    obs_np = obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs

    print(f"Running {args.num_episodes} episodes...")
    with torch.no_grad():
        while episode_count < args.num_episodes:
            actions, _ = model.predict(obs_np, deterministic=True)
            actions_t = torch.tensor(actions, dtype=torch.float32, device=args.device)
            obs, rewards, terminals, truncations, _ = env.step(actions_t)
            obs_np = obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs
            total_reward += rewards.sum().item()

            done = terminals | truncations
            if done.any():
                episode_count += 1
                print(f"Episode {episode_count} done")
                if episode_count < args.num_episodes:
                    obs, _ = env.reset()
                    obs_np = obs.cpu().numpy() if isinstance(obs, torch.Tensor) else obs

    print(f"\nDone. Total reward: {total_reward:.2f}, avg per episode: {total_reward / max(1, episode_count):.2f}")
    env.close()


if __name__ == "__main__":
    main()
