#!/usr/bin/env python3
"""Train a PPO agent on the quadcopter environment using PufferLib."""

import argparse
import torch
import pufferlib
import pufferlib.vector
from pufferlib import pufferl
from pufferlib.pufferl import WandbLogger
from pufferlib.emulation import GymnasiumPufferEnv
from drone_env import QuadcopterEnv
from export import export_weights
from time import time

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


def make_env_creator(**env_kwargs):
    """Create an environment creator function for PufferLib."""
    def env_creator(buf=None, seed=None):
        # Each environment instance simulates ONE quadcopter
        env = QuadcopterEnv(num_envs=1, **env_kwargs)
        return GymnasiumPufferEnv(env=env, buf=buf, seed=seed)
    return env_creator


def train(args):
    """Train a PPO agent using PufferLib."""

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create vectorized environment (PufferLib creates multiple env copies)
    vecenv = QuadcopterEnv(
        num_envs=args.num_envs,
        config_path=args.config_path,
        max_episode_length=args.max_episode_length,
        dt=args.dt,
        lin_vel_reward_scale=args.lin_vel_reward_scale,
        ang_vel_reward_scale=args.ang_vel_reward_scale,
        distance_to_goal_reward_scale=args.distance_to_goal_reward_scale,
        dynamics_randomization_delta=args.dynamics_randomization_delta,
        device=args.device,
    )

    # Create policy
    policy = Policy(vecenv.driver_env, hidden_size=args.hidden_size).to(args.device)

    print(f"Training PPO agent on {args.num_envs} parallel environments")
    print(f"Observation space: {vecenv.single_observation_space.shape}")
    print(f"Action space: {vecenv.single_action_space.shape}")
    print(f"Total timesteps: {args.total_timesteps}")

    # Load base config and override with command-line arguments
    config = pufferl.load_config('default')
    train_config = config['train']


    train_config['env'] = "l2f drone"

    # Sampling and batch parameters (matching SKRL config)
    # SKRL: rollouts=32, so batch_size = num_envs * rollouts
    train_config['total_timesteps'] = args.total_timesteps
    rollouts_multiplier = 32
    train_config['batch_size'] = args.num_envs * rollouts_multiplier  # SKRL rollouts=rollouts_multiplier
    train_config['bptt_horizon'] = 'auto'

    # SKRL: learning_epochs=8
    train_config['update_epochs'] = 8

    # SKRL: mini_batches=8, so minibatch_size = batch_size / 8
    # With num_envs=4096, batch=131072, minibatch=16384
    train_config['minibatch_size'] = (args.num_envs * rollouts_multiplier) // 8

    # PPO hyperparameters (matching SKRL config)
    train_config['gamma'] = 0.99              # SKRL: discount_factor
    train_config['gae_lambda'] = 0.95         # SKRL: lambda
    train_config['clip_coef'] = 0.2           # SKRL: ratio_clip
    train_config['vf_clip_coef'] = 0.2        # SKRL: value_clip
    train_config['vf_coef'] = 2.0             # SKRL: value_loss_scale
    train_config['ent_coef'] = 0.0            # SKRL: entropy_loss_scale
    train_config['max_grad_norm'] = 1.0       # SKRL: grad_norm_clip

    # Optimizer (SKRL uses Adam with lr=5e-4)
    train_config['optimizer'] = 'muon'
    train_config['learning_rate'] = 5.0e-04   # SKRL: learning_rate
    train_config['anneal_lr'] = True          # SKRL uses KLAdaptiveLR, we use cosine

    # Create trainer
    logger = WandbLogger({
        'wandb_project': 'puffer_raptor',
        'wandb_group': 'sim2sim',
        'tag': 'my_tag'
    })
    trainer = pufferl.PuffeRL(train_config, vecenv, policy, logger)

    start_time = time()
    # Training loop (5 minutes wall clock)
    try:
        while time() - start_time < 5*60:
            trainer.evaluate()
            logs = trainer.train()

            if trainer.global_step % args.print_interval == 0:
                print(f"Step: {trainer.global_step}/{args.total_timesteps}")
                if logs:
                    print(f"  Logs: {logs}")
    finally:
        # Save final model
        final_path = f"{args.exp_name}_final.pt"
        torch.save({
            'policy_state_dict': policy.state_dict(),
            'global_step': trainer.global_step,
        }, final_path)
        export_weights(policy, 'neural_network.c')
        print(f"Training complete! Saved final model to {final_path}")

        trainer.print_dashboard()
        trainer.close()


def main():
    parser = argparse.ArgumentParser(description="Train PPO agent on quadcopter environment")

    # Environment parameters
    parser.add_argument("--num-envs", type=int, default=4096, help="Number of parallel environments")
    parser.add_argument("--config-path", type=str, default="my_quad_parameters.json", help="Path to quadcopter config")
    parser.add_argument("--max-episode-length", type=int, default=2000, help="Maximum episode length")
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation timestep")
    parser.add_argument("--lin-vel-reward-scale", type=float, default=-0.0, help="Linear velocity reward scale")
    parser.add_argument("--ang-vel-reward-scale", type=float, default=-0.0, help="Angular velocity reward scale")
    parser.add_argument("--distance-to-goal-reward-scale", type=float, default=15.0, help="Distance to goal reward scale")
    parser.add_argument("--dynamics-randomization-delta", type=float, default=0.05, help="Dynamics randomization range")

    # Training parameters
    parser.add_argument("--hidden-size", type=int, default=32, help="Hidden layer size")
    parser.add_argument("--total-timesteps", type=int, default=100_000_000, help="Total training timesteps")

    # Logging and checkpointing
    parser.add_argument("--exp-name", type=str, default="quadcopter_ppo", help="Experiment name")
    parser.add_argument("--wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="puffer-raptor", help="W&B project name")
    parser.add_argument("--print-interval", type=int, default=10000, help="Print stats interval")
    parser.add_argument("--checkpoint-interval", type=int, default=100000, help="Checkpoint save interval (0 to disable)")

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
