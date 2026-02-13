#!/usr/bin/env python3
"""Train a PPO agent on the quadcopter environment using PufferLib."""

import argparse
import ast
import configparser
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime
from time import time

import torch
import pufferlib
from pufferlib import pufferl
from pufferlib.pufferl import WandbLogger
from drone_env import QuadcopterEnv
from export import export_weights


def load_pufferlib_config():
    """Load PufferLib's default.ini config without argparse (avoids sys.argv conflicts)."""
    puffer_dir = os.path.dirname(os.path.realpath(pufferlib.__file__))
    default_ini = os.path.join(puffer_dir, 'config', 'default.ini')

    p = configparser.ConfigParser()
    p.read(default_ini)

    config = defaultdict(dict)
    for section in p.sections():
        for key in p[section]:
            try:
                value = ast.literal_eval(p[section][key])
            except:
                value = p[section][key]

            # Build nested dict from dotted section names
            parts = section.split('.')
            d = config
            for part in parts:
                d = d.setdefault(part, {})
            d[key] = value

    # Flatten top-level sections (base keys go to root)
    result = defaultdict(dict)
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = value
        else:
            result[key] = value

    result['train']['use_rnn'] = False
    return result

class Policy(torch.nn.Module):
    """Simple MLP policy for continuous control."""
    def __init__(self, env, hidden_size):
        super().__init__()
        obs_shape = env.single_observation_space.shape[0]
        action_shape = env.single_action_space.shape[0]

        self.net = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(torch.nn.Linear(obs_shape, hidden_size)),
            torch.nn.ELU(),
            pufferlib.pytorch.layer_init(torch.nn.Linear(hidden_size, hidden_size)),
            torch.nn.ELU(),
        )
        self.action_mean = pufferlib.pytorch.layer_init(torch.nn.Linear(hidden_size, action_shape), std=0.01)
        self.action_logstd = torch.nn.Parameter(torch.zeros(1, action_shape))
        self.value_head = pufferlib.pytorch.layer_init(torch.nn.Linear(hidden_size, 1), std=1.0)

    def forward_eval(self, observations, state=None):
        """Forward pass during evaluation (returns Normal distribution and values)."""
        hidden = self.net(observations)
        action_mean = self.action_mean(hidden)
        action_logstd = self.action_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        values = self.value_head(hidden)
        # Return Normal distribution for PufferLib's sample_logits
        action_dist = torch.distributions.Normal(action_mean, action_std)
        return action_dist, values

    def forward(self, observations, state=None):
        """Forward pass during training."""
        return self.forward_eval(observations, state)



def train(args, wandb_group=None):
    """Train a PPO agent using PufferLib. Returns mean episode reward."""

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create vectorized environment (PufferLib creates multiple env copies)
    vecenv = QuadcopterEnv(
        num_envs=args.num_envs,
        config_path=args.config_path,
        lin_vel_reward_scale=args.lin_vel_reward_scale,
        ang_vel_reward_scale=args.ang_vel_reward_scale,
        distance_to_goal_reward_scale=args.distance_to_goal_reward_scale,
        dynamics_randomization_delta=args.dynamics_randomization_delta,
        device=args.device,
        use_compile=True
    )

    # Create policy
    policy = Policy(vecenv.driver_env, hidden_size=args.hidden_size).to(args.device)

    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint_path = os.path.abspath(args.checkpoint)
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        policy.load_state_dict(checkpoint['policy_state_dict'])

    print(f"Training PPO agent on {args.num_envs} parallel environments")
    print(f"Observation space: {vecenv.single_observation_space.shape}")
    print(f"Action space: {vecenv.single_action_space.shape}")
    print(f"Total timesteps: {args.total_timesteps}")

    # Load base config and override with command-line arguments
    config = load_pufferlib_config()
    train_config = config['train']

    train_config['env'] = "l2f drone"

    # Sampling and batch parameters (matching SKRL config)
    # SKRL: rollouts=32, so batch_size = num_envs * rollouts
    train_config['total_timesteps'] = args.total_timesteps
    rollouts_multiplier = 32
    train_config['batch_size'] = args.num_envs * rollouts_multiplier
    train_config['bptt_horizon'] = 'auto'
    train_config['minibatch_size'] = (args.num_envs * rollouts_multiplier)

    # PPO hyperparameters from CLI args
    train_config['update_epochs'] = args.update_epochs
    train_config['gamma'] = args.gamma
    train_config['gae_lambda'] = args.gae_lambda
    train_config['clip_coef'] = args.clip_coef
    train_config['vf_clip_coef'] = 0.2
    train_config['vf_coef'] = args.vf_coef
    train_config['ent_coef'] = args.ent_coef
    train_config['max_grad_norm'] = 1.0

    # Optimizer
    train_config['optimizer'] = 'muon'
    train_config['learning_rate'] = args.learning_rate
    train_config['anneal_lr'] = True

    # Initialize wandb early to get run ID for log directory
    logger = None
    wandb_url = None
    if args.wandb:
        logger = WandbLogger({
            'wandb_project': args.wandb_project,
            'wandb_group': wandb_group or 'sim2sim',
            'tag': 'my_tag',
            **vars(args),
        })
        run_id = logger.run_id
        wandb_url = logger.wandb.run.get_url()
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create per-run log directory
    log_dir = os.path.join("logs", run_id)
    os.makedirs(log_dir, exist_ok=True)

    # Copy flight params and save config snapshot
    shutil.copy(args.config_path, os.path.join(log_dir, "flight_params.json"))
    with open(os.path.join(log_dir, "config.json"), 'w') as f:
        json.dump({
            "args": vars(args),
            "train_config": dict(train_config),
            "run_id": run_id,
            "wandb_url": wandb_url,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    # Upload flight params and local path to wandb
    if args.wandb:
        artifact = logger.wandb.Artifact(f'flight_params_{run_id}', type='config')
        artifact.add_file(args.config_path)
        logger.wandb.run.log_artifact(artifact)
        logger.wandb.config.update({'local_log_dir': os.path.abspath(log_dir)})
        if args.checkpoint:
            logger.wandb.config.update({'checkpoint': os.path.abspath(args.checkpoint)})

    print(f"Log directory: {log_dir}")

    # Create trainer
    trainer = pufferl.PuffeRL(train_config, vecenv, policy, logger)

    start_time = time()
    mean_reward = 0.0
    # Training loop (2 minutes wall clock)
    try:
        while time() - start_time < 2*60:
            trainer.evaluate()
            logs = trainer.train()

            if logs and 'environment/episode_reward_mean' in logs:
                mean_reward = logs['environment/episode_reward_mean']

            if trainer.global_step % args.print_interval == 0:
                print(f"Step: {trainer.global_step}/{args.total_timesteps}")
                if logs:
                    print(f"  Logs: {logs}")
    finally:
        # Save all artifacts to log directory
        final_path = os.path.join(log_dir, "model.pt")
        torch.save({
            'policy_state_dict': policy.state_dict(),
            'global_step': trainer.global_step,
        }, final_path)
        export_weights(
            policy,
            os.path.join(log_dir, "neural_network.c"),
            wandb_url=wandb_url,
            run_id=run_id,
        )
        print(f"Training complete! Saved artifacts to {log_dir}")
        print(f"Final mean episode reward: {mean_reward}")

        trainer.print_dashboard()
        trainer.close()

    return mean_reward


def run_sweep(args):
    """Run hyperparameter sweep using PufferLib's built-in sweep system."""
    import pufferlib.sweep

    # Sweeps require wandb for aggregate viewing
    args.wandb = True

    # Unique group name so all runs in this sweep are grouped together
    sweep_group = f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Sweep group: {sweep_group}")
    print(f"View aggregate results in wandb by filtering group = {sweep_group}")

    config = load_pufferlib_config()
    sweep_config = config.pop('sweep', {})
    method = sweep_config.pop('method', 'Protein')

    sweep_cls = getattr(pufferlib.sweep, method)
    sweep = sweep_cls(sweep_config)

    sweep_keys = [
        'learning_rate', 'gamma', 'gae_lambda', 'clip_coef',
        'vf_coef', 'ent_coef', 'update_epochs',
    ]

    for i in range(args.max_sweep_runs):
        print(f"\n=== Sweep Run {i+1}/{args.max_sweep_runs} ===")

        sweep.suggest(config)

        # Override args with suggested train parameters
        for key in sweep_keys:
            if key in config['train']:
                setattr(args, key.replace('-', '_'), config['train'][key])

        # Override env parameters if suggested
        env_config = config.get('env', {})
        for key in ['distance_to_goal_reward_scale', 'dynamics_randomization_delta']:
            if key in env_config:
                setattr(args, key.replace('-', '_'), env_config[key])

        start = time()
        score = train(args, wandb_group=sweep_group)
        duration = time() - start

        print(f"Run {i+1} score: {score:.2f}, duration: {duration:.0f}s")
        sweep.observe(config, score, duration)


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on quadcopter environment")

    # Environment parameters
    parser.add_argument("--num-envs", type=int, default=2048, help="Number of parallel environments")
    parser.add_argument("--config-path", type=str, default="meteor75_parameters.json", help="Path to quadcopter config")
    parser.add_argument("--max-episode-length", type=int, default=2000, help="Maximum episode length")
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation timestep")
    parser.add_argument("--lin-vel-reward-scale", type=float, default=-0.0, help="Linear velocity reward scale")
    parser.add_argument("--ang-vel-reward-scale", type=float, default=-0.0, help="Angular velocity reward scale")
    parser.add_argument("--distance-to-goal-reward-scale", type=float, default=15.0, help="Distance to goal reward scale")
    parser.add_argument("--dynamics-randomization-delta", type=float, default=0.1, help="Dynamics randomization range")

    # Training parameters
    parser.add_argument("--hidden-size", type=int, default=32, help="Hidden layer size")
    parser.add_argument("--total-timesteps", type=int, default=100_000_000, help="Total training timesteps")

    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=5.0e-04, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-coef", type=float, default=0.2, help="PPO clip coefficient")
    parser.add_argument("--vf-coef", type=float, default=2.0, help="Value function loss coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.0, help="Entropy loss coefficient")
    parser.add_argument("--update-epochs", type=int, default=8, help="PPO update epochs per rollout")

    # Logging and checkpointing
    parser.add_argument("--exp-name", type=str, default="quadcopter_ppo", help="Experiment name")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="puffer-raptor", help="W&B project name")
    parser.add_argument("--print-interval", type=int, default=10000, help="Print stats interval")
    parser.add_argument("--checkpoint-interval", type=int, default=100000, help="Checkpoint save interval (0 to disable)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file to resume from")

    # Sweep
    parser.add_argument("--run-sweep", action="store_true", help="Run hyperparameter sweep")
    parser.add_argument("--max-sweep-runs", type=int, default=50, help="Number of sweep runs")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")

    args = parser.parse_args()

    if args.run_sweep:
        run_sweep(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
