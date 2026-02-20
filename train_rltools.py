from rltools import SAC
import argparse
import os
import shutil
import time
from datetime import datetime

import gymnasium as gym
import numpy as np
from drone_env import QuadcopterEnv


class _Spec:
    """Minimal env spec so rltools get_time_limit() can read max_episode_steps."""
    def __init__(self, max_episode_steps):
        self.max_episode_steps = max_episode_steps


class _SingleEnvWrapper:
    """Wrap QuadcopterEnv(num_envs=1) to present a single-instance Gym interface.

    rltools expects unbatched (obs_dim,) arrays and scalar reward/done, but
    QuadcopterEnv always returns (num_envs, ...) tensors.
    """
    def __init__(self, env):
        self._env = env
        self.observation_space = env.single_observation_space
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.spec = _Spec(env.max_episode_length)

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        return obs[0].numpy(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        return (
            obs[0].numpy(),
            float(reward[0]),
            bool(terminated[0]),
            bool(truncated[0]),
            info,
        )


def make_single_env(config_path):
    """Return a factory that creates a single Gymnasium-compatible QuadcopterEnv."""
    def env_factory():
        env = QuadcopterEnv(num_envs=1, config_path=config_path, device="cpu")
        return _SingleEnvWrapper(env)
    return env_factory


def train(args):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs_rltools", run_id)
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy(args.config_path, os.path.join(log_dir, "flight_params.json"))

    env_factory = make_single_env(args.config_path)

    print("Compiling rltools SAC...")
    sac = SAC(env_factory)
    state = sac.State(args.seed)

    print(f"Training | log_dir={log_dir}")
    start_time = time.time()
    finished = False
    while not finished and time.time() - start_time < args.train_minutes * 60:
        finished = state.step()

    checkpoint_path = os.path.join(log_dir, "checkpoint.h")
    with open(checkpoint_path, "w") as f:
        f.write(state.export_policy())
    print(f"Saved checkpoint to {checkpoint_path}")


def main():
    p = argparse.ArgumentParser(description="Train SAC via rltools on the quadcopter env")
    p.add_argument("--config-path", type=str, default="meteor75_parameters.json")
    p.add_argument("--train-minutes", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0xf00d)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
