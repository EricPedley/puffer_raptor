#!/usr/bin/env python3
"""Load the latest checkpoint from train_ppo.py and run a rollout with rendering."""

import argparse
import glob
import os
import torch
from pathlib import Path

from drone_env import QuadcopterEnv
from train_ppo import Policy, load_config


def find_latest_checkpoint(exp_name: str = "quadcopter_ppo") -> str:
    """Find the latest checkpoint file for the given experiment name."""
    checkpoint_pattern = f"{exp_name}*.pt"
    checkpoints = glob.glob(checkpoint_pattern)

    if not checkpoints:
        # Try the final checkpoint
        final_checkpoint = f"{exp_name}_final.pt"
        if os.path.exists(final_checkpoint):
            return final_checkpoint
        raise FileNotFoundError(f"No checkpoints found matching pattern: {checkpoint_pattern}")

    # Sort by modification time and return the latest
    latest = max(checkpoints, key=os.path.getmtime)
    return latest


def load_checkpoint(checkpoint_path: str, policy: torch.nn.Module, device: str):
    """Load checkpoint into policy."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    global_step = checkpoint.get('global_step', 0)
    return global_step


def run_rollout(
    policy: torch.nn.Module,
    env: QuadcopterEnv,
    num_episodes: int = 1,
    max_steps: int = None,
    device: str = "cuda",
):
    """Run a rollout with the given policy."""
    policy.eval()

    total_reward = 0.0
    episode_count = 0

    print(f"Running {num_episodes} rollout episodes with rendering...")

    obs, _ = env.reset()
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    state = {}
    with torch.no_grad():
        while True:
            # Get action from policy
            action_dist, value = policy.forward_eval(obs, state)
            actions = action_dist.mean  # Use deterministic actions (mean of distribution)

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
                # Reset LSTM state and environments
                state = {}
                obs, _ = env.reset()
                obs = torch.tensor(obs, dtype=torch.float32, device=device)

    avg_reward = total_reward / max(1, episode_count)
    print(f"\nRollout complete!")
    print(f"Episodes: {episode_count}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per episode: {avg_reward:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run rollout with rendering using latest checkpoint")

    parser.add_argument("--exp-name", type=str, default="quadcopter_ppo", help="Experiment name")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint (auto-find latest if not specified)")
    parser.add_argument("--latest", action="store_true", help="Use model.pt from the most recent logs/ subfolder")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--config-ini", type=str, default="drone.ini", help="Path to drone.ini config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--config-path", type=str, default="meteor75_parameters.json", help="Path to quadcopter config")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Find checkpoint
    if args.latest:
        logs_dir = Path(__file__).parent / "logs"
        subdirs = [d for d in logs_dir.iterdir() if d.is_dir()]
        if not subdirs:
            raise FileNotFoundError("No subdirectories found in logs/")
        latest_dir = max(subdirs, key=lambda d: d.stat().st_mtime)
        checkpoint_path = str(latest_dir / "model.pt")
    elif args.checkpoint is None:
        checkpoint_path = find_latest_checkpoint(args.exp_name)
    else:
        checkpoint_path = args.checkpoint

    print(f"Loading checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Create environment with rendering
    env = QuadcopterEnv(
        num_envs=args.num_envs,
        config_path=args.config_path,
        device=args.device,
        render_mode="human",  # Enable rendering
    )

    # Read policy sizes from ini
    ini_config = load_config(args.config_ini)
    policy_config = ini_config.get('policy', {})
    hidden_size = policy_config.get('linear_size', 64)
    rnn_hidden_size = policy_config.get('lstm_size', 16)

    # Create policy
    policy = Policy(env, hidden_size=hidden_size, rnn_hidden_size=rnn_hidden_size).to(args.device)

    # Load checkpoint
    global_step = load_checkpoint(checkpoint_path, policy, args.device)
    print(f"Loaded checkpoint from step {global_step}")

    # Run rollout
    run_rollout(
        policy=policy,
        env=env,
        num_episodes=args.num_episodes,
        device=args.device,
    )

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
