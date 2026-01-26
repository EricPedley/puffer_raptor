#!/usr/bin/env python3
"""Test the quadcopter environment without isaaclab dependencies."""

import torch
from drone_env import QuadcopterEnv


def test_environment():
    """Test basic environment functionality."""
    print("Creating quadcopter environment...")
    env = QuadcopterEnv(
        num_envs=4,
        config_path="my_quad_parameters.json",
        max_episode_length=100,
        device="cpu"
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Reset environment
    print("\nResetting environment...")
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation:\n{obs}")

    # Run a few steps
    print("\nRunning 10 random steps...")
    total_rewards = torch.zeros(env.num_envs)

    for step in range(10):
        # Random actions
        actions = torch.rand(env.num_envs, 4) * 2 - 1  # Range [-1, 1]

        # Step environment
        obs, rewards, terminated, truncated, info = env.step(actions)

        total_rewards += rewards

        print(f"Step {step + 1}:")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Rewards: {rewards}")
        print(f"  Terminated: {terminated.sum().item()} envs")
        print(f"  Truncated: {truncated.sum().item()} envs")

    print(f"\nTotal rewards over 10 steps: {total_rewards}")
    print("\nEnvironment test passed!")


if __name__ == "__main__":
    test_environment()
