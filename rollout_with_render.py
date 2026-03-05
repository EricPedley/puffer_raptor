#!/usr/bin/env python3
"""Load a checkpoint or the Raptor foundation policy and run a rollout with rendering."""

import argparse
import glob
import os
import torch
from pathlib import Path

from drone_env import QuadcopterEnv
from race_env import QuadcopterRaceEnv
# from train_ppo import Policy
from train_race import Policy
from train_sac import Actor
import foundation_policy

from racing_env_example.gate_maps import (
    gate_positions as _hard_pos,
    gate_yaws as hard_yaw,
    racetrack_start as _hard_start,
    positions_with_extr_gate as _easy_pos,
    yaws_with_extra_gate as easy_yaw,
    easy_start as _easy_start,
)

def _zdown_to_zup(pos):
    p = pos.copy()
    p[:, 2] = -p[:, 2]
    return p

easy_pos   = _zdown_to_zup(_easy_pos)
hard_pos   = _zdown_to_zup(_hard_pos)
easy_start = _easy_start.copy(); easy_start[2] = -easy_start[2]
hard_start = _hard_start.copy(); hard_start[2] = -_hard_start[2]


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
    policy.load_state_dict(checkpoint)
    global_step = checkpoint.get("global_step", 0)
    return policy, is_sac, global_step


def get_action_torch(policy: torch.nn.Module, obs: torch.Tensor, is_sac: bool) -> torch.Tensor:
    """Get deterministic action from a torch policy (PPO or SAC)."""
    if is_sac:
        _, _, mean = policy.get_action(obs)
        return mean
    else:
        action_dist, _ = policy.forward_eval(obs)
        return action_dist.mean


def build_raptor_obs(env: QuadcopterEnv) -> torch.Tensor:
    """Build 22D rl-tools observation from the env's internal state.

    Raptor expects: pos_error_world (3) + rotation_matrix (9) +
                    velocity_world (3) + angular_velocity_body (3) + last_action (4) = 22

    The env's 19D obs uses body-frame quantities that can't be inverted,
    so we read directly from the env's state tensors.
    """
    # Position error (world frame)
    position_error = env._position - env._desired_pos_w

    # Rotation matrix from quaternion (flattened 3x3)
    q = env._quaternion
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rot_matrix = torch.stack([
        1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y),
        2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y),
    ], dim=-1)

    # Velocity (world frame)
    velocity = env._velocity

    # Angular velocity (body frame)
    angular_velocity = env._angular_velocity

    # Last action (clamped [-1, 1])
    last_action = env._actions

    return torch.cat([
        position_error,       # 3
        rot_matrix,           # 9
        velocity,             # 3
        angular_velocity,     # 3
        last_action,          # 4
    ], dim=-1)


def get_action_raptor(raptor_model, env: QuadcopterEnv) -> torch.Tensor:
    """Get action from the Raptor foundation policy.

    Builds the 22D rl-tools obs from env state, runs through the numpy-based
    GRU model via evaluate_step, and returns a torch tensor.
    """
    obs_22d = build_raptor_obs(env)
    obs_np = obs_22d.cpu().numpy()
    actions_np = raptor_model.evaluate_step(obs_np)
    return torch.tensor(actions_np, dtype=obs_22d.dtype, device=obs_22d.device)


def run_rollout(
    policy,
    env: QuadcopterEnv,
    is_raptor: bool,
    is_sac: bool = False,
    num_episodes: int = 1,
    device: str = "cuda",
):
    """Run a rollout with the given policy."""
    if not is_raptor:
        policy.eval()

    total_reward = 0.0
    episode_count = 0

    print(f"Running {num_episodes} rollout episodes with rendering...")

    obs, _ = env.reset()
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32, device=device)
    if is_raptor:
        policy.reset()

    with torch.no_grad():
        while True:
            if is_raptor:
                actions = get_action_raptor(policy, env)
            else:
                actions = get_action_torch(policy, obs, is_sac)

            # Step environment
            obs, rewards, terminals, truncations, infos = env.step(actions)
            if not isinstance(obs, torch.Tensor):
                obs = torch.tensor(obs, dtype=torch.float32, device=device)

            total_reward += rewards.sum().item()

            # Check if episode is done
            done = terminals | truncations
            if done.any():
                episode_count += 1
                if episode_count >= num_episodes:
                    break
                obs, _ = env.reset()
                if not isinstance(obs, torch.Tensor):
                    obs = torch.tensor(obs, dtype=torch.float32, device=device)
                if is_raptor:
                    policy.reset()

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
    parser.add_argument("--raptor", action="store_true", help="Use the Raptor foundation policy instead of a checkpoint")
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to run")
    parser.add_argument("--hidden-size", type=int, default=64, help="Hidden layer size of policy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--config-path", type=str, default="meteor75_parameters.json", help="Path to quadcopter config")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel environments")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)

    # Create environment with rendering
    env = QuadcopterRaceEnv(
        gate_positions=easy_pos,
        gate_yaws =easy_yaw,
        start_position=easy_start,
        num_envs=args.num_envs,
        config_path=args.config_path,
        device=args.device,
        render_mode="human",
        # dt=1/500,
        # max_episode_length=500*20,
        # discretize_obs=False
    )

    if args.raptor:
        print("Loading Raptor foundation policy...")
        policy = foundation_policy.Raptor()
        print(f"Model: {policy.description()}")
        run_rollout(
            policy=policy,
            env=env,
            is_raptor=True,
            num_episodes=args.num_episodes,
            device=args.device,
        )
    else:
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

        policy, is_sac, global_step = load_policy(checkpoint_path, env, args.hidden_size, args.device)
        print(f"Loaded checkpoint from step {global_step}")

        run_rollout(
            policy=policy,
            env=env,
            is_raptor=False,
            is_sac=is_sac,
            num_episodes=args.num_episodes,
            device=args.device,
        )

    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
