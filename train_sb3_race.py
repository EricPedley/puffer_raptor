#!/usr/bin/env python3
"""Train a PPO agent on the race env using Stable Baselines 3, matching racing_env_example/train.py."""

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecEnv, VecMonitor

from race_env import QuadcopterRaceEnv
from train_race import easy_pos, easy_yaw, easy_start, hard_pos, hard_yaw, hard_start


class RaceVecEnv(VecEnv):
    """Thin SB3 VecEnv wrapper around QuadcopterRaceEnv."""

    def __init__(self, env: QuadcopterRaceEnv):
        self._env = env
        obs_shape = env.single_observation_space.shape
        act_shape = env.single_action_space.shape
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        action_space = spaces.Box(low=-1.0, high=1.0, shape=act_shape, dtype=np.float32)
        super().__init__(env.num_envs, observation_space, action_space)
        self._actions = None

    def reset(self):
        obs, _ = self._env.reset()
        return obs.cpu().numpy()

    def step_async(self, actions: np.ndarray):
        self._actions = actions

    def step_wait(self):
        actions = torch.tensor(self._actions, dtype=torch.float32, device=self._env.device)
        obs, rewards, terminals, truncations, infos = self._env.step(actions)
        dones = (terminals | truncations).cpu().numpy()
        obs_np = obs.cpu().numpy()
        rew_np = rewards.cpu().numpy()
        # SB3 expects a list of info dicts, one per env
        info_list = [{} for _ in range(self._env.num_envs)]
        for i in range(self._env.num_envs):
            if dones[i]:
                info_list[i]["terminal_observation"] = obs_np[i]
        return obs_np, rew_np, dones, info_list

    def close(self):
        self._env.close()

    def get_attr(self, attr_name, indices=None):
        raise AttributeError(attr_name)

    def set_attr(self, attr_name, value, indices=None):
        pass

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs

    def seed(self, seed=None):
        return [None] * self.num_envs


def make_env(gate_pos, gate_yaw, start_pos, num_envs=100, config_path="meteor75_parameters.json"):
    inner = QuadcopterRaceEnv(
        gate_positions=gate_pos,
        gate_yaws=gate_yaw,
        start_position=start_pos,
        num_envs=num_envs,
        config_path=config_path,
        initialize_at_random_gates=True,
    )
    return VecMonitor(RaceVecEnv(inner))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=100)
    parser.add_argument("--config-path", type=str, default="meteor75_parameters.json")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--total-timesteps", type=int, default=200_000_000)
    parser.add_argument("--track", type=str, default="easy", choices=["easy", "hard"])
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    if args.track == "easy":
        gate_pos, gate_yaw, start_pos = easy_pos, easy_yaw, easy_start
    else:
        gate_pos, gate_yaw, start_pos = hard_pos, hard_yaw, hard_start

    env = make_env(gate_pos, gate_yaw, start_pos, num_envs=args.num_envs, config_path=args.config_path)

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[dict(pi=[args.hidden_size, args.hidden_size], vf=[args.hidden_size, args.hidden_size])],
        log_std_init=0,
    )

    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model = PPO.load(args.checkpoint, env=env)
        model.ent_coef = 0.0
    else:
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=1000,
            batch_size=5000,
            n_epochs=10,
            gamma=0.999,
            ent_coef=0.0,
            tensorboard_log="logs/sb3_race",
        )

    class LogDirCheckpointCallback(CheckpointCallback):
        def _init_callback(self):
            self.save_path = self.model.logger.dir
            super()._init_callback()

    checkpoint_callback = LogDirCheckpointCallback(
        save_freq=max(1_000_000 // args.num_envs, 1),
        save_path="logs/sb3_race",  # overridden in _init_callback
        name_prefix=f"sb3_race_{args.track}",
        verbose=2,
    )

    print(model.policy)
    model.learn(
        total_timesteps=args.total_timesteps,
        reset_num_timesteps=True,
        tb_log_name="ppo",
        callback=checkpoint_callback,
    )
