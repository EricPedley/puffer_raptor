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


def load_config(drone_ini_path='drone.ini'):
    """Load pufferlib default.ini as base, then override with drone.ini."""
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
            parts = section.split('.')
            d = config
            for part in parts:
                d = d.setdefault(part, {})
            d[key] = value

    result = defaultdict(dict)
    for key, value in config.items():
        if isinstance(value, dict):
            result[key] = value
        else:
            result[key] = value

    # Override with drone.ini
    dp = configparser.ConfigParser()
    dp.read(drone_ini_path)
    for section in dp.sections():
        for key in dp[section]:
            try:
                value = ast.literal_eval(dp[section][key])
            except:
                value = dp[section][key]
            parts = section.split('.')
            d = result
            for part in parts:
                d = d.setdefault(part, {})
            d[key] = value

    return result


class Policy(torch.nn.Module):
    """LSTM policy for continuous control."""
    def __init__(self, env, hidden_size, rnn_hidden_size=16):
        super().__init__()
        obs_shape = env.single_observation_space.shape[0]
        action_shape = env.single_action_space.shape[0]
        self.hidden_size = rnn_hidden_size

        self.encoder = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(torch.nn.Linear(obs_shape, hidden_size)),
            torch.nn.ELU(),
        )
        self.lstm = torch.nn.LSTM(hidden_size, rnn_hidden_size, batch_first=True)
        self.decoder = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(torch.nn.Linear(rnn_hidden_size, hidden_size)),
            torch.nn.ELU(),
        )
        self.action_mean = pufferlib.pytorch.layer_init(torch.nn.Linear(hidden_size, action_shape), std=0.01)
        self.action_logstd = torch.nn.Parameter(torch.zeros(1, action_shape))
        self.value_head = pufferlib.pytorch.layer_init(torch.nn.Linear(hidden_size, 1), std=1.0)

    def forward_eval(self, observations, state=None):
        """Forward pass during evaluation."""
        hidden = self.encoder(observations)
        hidden = hidden.unsqueeze(1)  # (B, 1, H) for single-step LSTM

        lstm_h = state.get('lstm_h') if state else None
        lstm_c = state.get('lstm_c') if state else None
        if lstm_h is not None:
            lstm_h = lstm_h.unsqueeze(0)  # (1, B, rnn_H)
            lstm_c = lstm_c.unsqueeze(0)
            hidden, (lstm_h, lstm_c) = self.lstm(hidden, (lstm_h, lstm_c))
        else:
            hidden, (lstm_h, lstm_c) = self.lstm(hidden)

        state['lstm_h'] = lstm_h.squeeze(0)
        state['lstm_c'] = lstm_c.squeeze(0)

        hidden = hidden.squeeze(1)  # (B, rnn_H)
        hidden = self.decoder(hidden)
        action_mean = self.action_mean(hidden)
        action_logstd = self.action_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        values = self.value_head(hidden)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        return action_dist, values

    def forward(self, observations, state=None):
        """Forward pass during training (handles bptt_horizon sequence dimension)."""
        # observations: (segments, bptt_horizon, obs_dim)
        B, T, _ = observations.shape
        hidden = self.encoder(observations)  # (B, T, H)
        hidden, _ = self.lstm(hidden)  # (B, T, rnn_H)
        hidden = hidden.reshape(B * T, -1)  # (B*T, rnn_H)
        hidden = self.decoder(hidden)
        action_mean = self.action_mean(hidden)
        action_logstd = self.action_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        values = self.value_head(hidden)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        return action_dist, values



def train(args, wandb_group=None):
    """Train a PPO agent using PufferLib. Returns mean episode reward."""

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load config from drone.ini (with pufferlib defaults as base)
    config = load_config(args.config_ini)
    train_config = config['train']
    env_config = config.get('env', {})
    policy_config = config.get('policy', {})

    # Create vectorized environment
    vecenv = QuadcopterEnv(
        num_envs=args.num_envs,
        config_path=args.config_path,
        dynamics_randomization_delta=args.dynamics_randomization_delta,
        device=args.device,
        use_compile=True
    )

    # Create policy from ini config
    hidden_size = policy_config.get('linear_size', 64)
    rnn_hidden_size = policy_config.get('lstm_size', 16)
    policy = Policy(vecenv.driver_env, hidden_size=hidden_size, rnn_hidden_size=rnn_hidden_size).to(args.device)

    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint_path = os.path.abspath(args.checkpoint)
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        policy.load_state_dict(checkpoint['policy_state_dict'])

    print(f"Training PPO agent on {args.num_envs} parallel environments")
    print(f"Observation space: {vecenv.single_observation_space.shape}")
    print(f"Action space: {vecenv.single_action_space.shape}")
    print(f"Policy: linear_size={hidden_size}, lstm_size={rnn_hidden_size}")
    print(f"Total timesteps: {train_config.get('total_timesteps', args.total_timesteps)}")

    # Set env name for dashboard
    train_config['env'] = "l2f drone"

    # Override total_timesteps from CLI if provided
    if args.total_timesteps is not None:
        train_config['total_timesteps'] = args.total_timesteps

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
    shutil.copy(args.config_ini, os.path.join(log_dir, "drone.ini"))
    with open(os.path.join(log_dir, "config.json"), 'w') as f:
        json.dump({
            "args": vars(args),
            "train_config": dict(train_config),
            "env_config": dict(env_config),
            "policy_config": dict(policy_config),
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
    try:
        while trainer.global_step < train_config['total_timesteps']:
            trainer.evaluate()
            logs = trainer.train()

            if logs and 'environment/episode_reward_mean' in logs:
                mean_reward = logs['environment/episode_reward_mean']

            if trainer.global_step % args.print_interval == 0:
                print(f"Step: {trainer.global_step}/{train_config['total_timesteps']}")
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

    config = load_config(args.config_ini)
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

        start = time()
        score = train(args, wandb_group=sweep_group)
        duration = time() - start

        print(f"Run {i+1} score: {score:.2f}, duration: {duration:.0f}s")
        sweep.observe(config, score, duration)


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on quadcopter environment")

    # Config file
    parser.add_argument("--config-ini", type=str, default="drone.ini", help="Path to drone.ini config file")

    # Environment parameters (CLI overrides)
    parser.add_argument("--num-envs", type=int, default=8*16*64, help="Number of parallel environments")
    parser.add_argument("--config-path", type=str, default="meteor75_parameters.json", help="Path to quadcopter config")
    parser.add_argument("--dynamics-randomization-delta", type=float, default=0.0, help="Dynamics randomization range")

    # Training parameters (CLI overrides, ini values take precedence for most hparams)
    parser.add_argument("--total-timesteps", type=int, default=None, help="Total training timesteps (overrides ini)")

    # Logging and checkpointing
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="puffer-raptor", help="W&B project name")
    parser.add_argument("--print-interval", type=int, default=10000, help="Print stats interval")
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
