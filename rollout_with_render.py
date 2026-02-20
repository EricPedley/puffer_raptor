#!/usr/bin/env python3
"""Load the latest checkpoint from train_ppo.py and run a rollout with rendering."""

import argparse
import glob
import os
import torch
from pathlib import Path

from drone_env import QuadcopterEnv
from train_ppo import Policy
from train_sac import Actor


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


def load_policy(checkpoint_path: str, env: QuadcopterEnv, hidden_size: int, device: str):
    """Detect checkpoint type from keys and load the appropriate policy class."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    is_sac = "qf1" in checkpoint
    if is_sac:
        print("Detected SAC checkpoint")
        policy = Actor(
            obs_dim=env.single_observation_space.shape[0],
            action_dim=env.single_action_space.shape[0],
            hidden_size=hidden_size,
        ).to(device)
    else:
        print("Detected PPO checkpoint")
        policy = Policy(env, hidden_size=hidden_size).to(device)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    global_step = checkpoint.get("global_step", 0)
    return policy, is_sac, global_step


def get_action(policy: torch.nn.Module, obs: torch.Tensor, is_sac: bool) -> torch.Tensor:
    """Get deterministic action from either policy type."""
    if is_sac:
        _, _, mean = policy.get_action(obs)
        return mean
    else:
        action_dist, _ = policy.forward_eval(obs)
        return action_dist.mean


def run_rollout(
    policy: torch.nn.Module,
    env: QuadcopterEnv,
    is_sac: bool,
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

    with torch.no_grad():
        while True:
            actions = get_action(policy, obs, is_sac)

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
    parser.add_argument("--hidden-size", type=int, default=32, help="Hidden layer size of policy")
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

    # Load checkpoint (auto-detects PPO vs SAC from keys)
    policy, is_sac, global_step = load_policy(checkpoint_path, env, args.hidden_size, args.device)
    print(f"Loaded checkpoint from step {global_step}")

    # Run rollout
    run_rollout(
        policy=policy,
        env=env,
        is_sac=is_sac,
        num_episodes=args.num_episodes,
        device=args.device,
    )

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
