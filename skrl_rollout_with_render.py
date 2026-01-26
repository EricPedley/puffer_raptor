#!/usr/bin/env python3
"""Load the latest checkpoint from skrl training and run a rollout with rendering."""

import argparse
import glob
import os
import torch
import yaml
from pathlib import Path
from drone_env import QuadcopterEnv
from skrl.utils.runner.torch import Runner
from skrl_train import PufferEnvSKRLWrapper

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_rollout(
    runner: Runner,
    env: QuadcopterEnv,
    num_episodes: int = 1,
    max_steps: int = None,
    device: str = "cuda",
):
    """Run a rollout with the given skrl runner."""
    if max_steps is None:
        max_steps = env.max_episode_length

    runner.agent.policy.eval()

    total_reward = 0.0
    episode_count = 0

    print(f"Running {num_episodes} rollout episodes with rendering...")

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    with torch.no_grad():
        for step in range(max_steps * num_episodes):
            # Get action from policy using the act method with required arguments
            actions, _, _ = runner.agent.act(obs, 0, 0)

            # Step environment
            obs, rewards, terminals, truncations, infos = env.step(actions)
            obs = torch.tensor(obs, dtype=torch.float32, device=device)

            total_reward += rewards.sum().item()

            # Check if episode is done
            done = terminals | truncations
            if done.any():
                episode_count += 1
                if episode_count >= num_episodes:
                    break
                # Reset only the done environments
                obs, _ = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32, device=device)

            # Print progress
            if (step + 1) % 100 == 0:
                print(f"Step: {step + 1}")

    avg_reward = total_reward / max(1, episode_count)
    print(f"\nRollout complete!")
    print(f"Episodes: {episode_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per episode: {avg_reward:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run rollout with rendering using skrl checkpoint")

    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--config-path", type=str, default="skrl_ppo_config.yaml", help="Path to skrl config")
    parser.add_argument("--env-config-path", type=str, default="my_quad_parameters.json", help="Path to quadcopter config")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Find checkpoint
    checkpoint_path = args.checkpoint

    print(f"Loading checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load agent configuration
    agent_cfg = load_config(args.config_path)

    # Create environment with rendering
    env = QuadcopterEnv(
        num_envs=args.num_envs,
        config_path=args.env_config_path,
        max_episode_length=args.max_steps,
        device=args.device,
        render_mode="human",  # Enable rendering
    )

    # Create skrl runner
    runner = Runner(PufferEnvSKRLWrapper(env), agent_cfg)

    # Load checkpoint
    runner.agent.load(checkpoint_path)
    print(f"Loaded checkpoint from: {checkpoint_path}")

    # Run rollout
    run_rollout(
        runner=runner,
        env=env,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        device=args.device,
    )

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
