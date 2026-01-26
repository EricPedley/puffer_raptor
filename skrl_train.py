"""
Script to train RL agent with skrl on pufferlib-compatible environments.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

import argparse
import os
import random
import yaml
from datetime import datetime

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

from skrl.utils.runner.torch import Runner
from skrl.envs.wrappers.torch import Wrapper
import pufferlib
import gymnasium
import torch
from typing import Tuple, Any
from drone_env import QuadcopterEnv

class PufferEnvSKRLWrapper(Wrapper):
    def __init__(self, env: pufferlib.PufferEnv):
        super().__init__(env)
        self.env=env

    def reset(self) -> Tuple[torch.Tensor, Any]:
        """Reset the environment

        :raises NotImplementedError: Not implemented

        :return: Observation, info
        :rtype: torch.Tensor and any other info
        """
        return self.env.reset()

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Perform a step in the environment

        :param actions: The actions to perform
        :type actions: torch.Tensor

        :raises NotImplementedError: Not implemented

        :return: Observation, reward, terminated, truncated, info
        :rtype: tuple of torch.Tensor and any other info
        """
        return self.env.step(actions)

    def state(self) -> torch.Tensor:
        """Get the environment state

        :raises NotImplementedError: Not implemented

        :return: State
        :rtype: torch.Tensor
        """
        return None

    def render(self, *args, **kwargs) -> Any:
        """Render the environment

        :raises NotImplementedError: Not implemented

        :return: Any value from the wrapped environment
        :rtype: any
        """
        return

    def close(self) -> None:
        """Close the environment

        :raises NotImplementedError: Not implemented
        """
        return self.env.close()


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config_path: str, config: dict) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def main(args_cli):
    """Train with skrl agent on pufferlib environment."""

    # Load agent configuration
    agent_cfg = load_config("skrl_ppo_config.yaml")

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent seed from command line
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]

    # Determine device
    device = "cuda" if args_cli.device == "cuda" else "cpu"

    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]

    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_torch"
    print(f"Exact experiment name requested from command line: {log_dir}")

    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'

    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    save_config(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = args_cli.checkpoint if args_cli.checkpoint else None

    # create pufferlib environment
    num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    env = QuadcopterEnv(
        num_envs=num_envs,
        device=device,
        render_mode='human'
    )
    skrl_env = PufferEnvSKRLWrapper(env)

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(skrl_env, agent_cfg)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # run training
    runner.run()

    # close the environment
    skrl_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent with skrl on pufferlib environment.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
    parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for training.",
    )

    args = parser.parse_args()
    main(args)
