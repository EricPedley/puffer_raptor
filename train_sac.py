#!/usr/bin/env python3
"""SAC on the quadcopter env, reproducing the rl-tools L2F SAC setup.

rl-tools reference: src/rl/environments/l2f/sac/parameters.h
Key choices matched:
  - hidden=32, TANH activation (ACTOR_HIDDEN_DIM=32, TANH)
  - gamma=0.99, batch_size=64, lr=1e-3
  - N_WARMUP_STEPS=1000
  - autotune entropy (ENTROPY_BONUS=true)
  - actor trained every 2 critic updates (ACTOR_TRAINING_INTERVAL = 2*CRITIC_TRAINING_INTERVAL)
  - reward: scale=1, constant=0.5, orientation_weight=0.1, term_position=1m, init_position=0.5m
"""
import argparse
import os
import random
import shutil
import time
from datetime import datetime

import numpy as np
import pufferlib.pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from drone_env import QuadcopterEnv
from export import export_weights


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class SoftQNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, a):
        return self.net(torch.cat([x, a], dim=-1))


class Actor(nn.Module):
    """Same architecture as train_ppo.py Policy — ELU MLP + action_mean + global action_logstd.
    Structured so export_weights() works directly (expects .net and .action_mean)."""
    def __init__(self, obs_dim, action_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(obs_dim, hidden_size)),
            nn.ELU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ELU(),
        )
        self.action_mean   = pufferlib.pytorch.layer_init(nn.Linear(hidden_size, action_dim), std=0.01)
        self.action_logstd = nn.Parameter(torch.zeros(1, action_dim))

    def get_action(self, x):
        hidden  = self.net(x)
        mean    = self.action_mean(hidden)
        log_std = self.action_logstd.expand_as(mean).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()
        normal  = torch.distributions.Normal(mean, std)
        x_t     = normal.rsample()
        y_t     = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return y_t, log_prob, torch.tanh(mean)

class Actor_Old(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
        )
        self.fc_mean = nn.Linear(hidden_size, action_dim)
        self.fc_logstd = nn.Linear(hidden_size, action_dim)

    def forward(self, x):
        h = self.net(x)
        mean = self.fc_mean(h)
        log_std = torch.tanh(self.fc_logstd(h))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        # action is tanh-squashed; drone_env clamps to [-1,1] anyway
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return y_t, log_prob, torch.tanh(mean)


# ── GPU-resident replay buffer ─────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim, device):
        self.cap = capacity
        self.device = device
        self.pos = 0
        self._size = 0
        self.obs      = torch.zeros(capacity, obs_dim,    device=device)
        self.next_obs = torch.zeros(capacity, obs_dim,    device=device)
        self.actions  = torch.zeros(capacity, action_dim, device=device)
        self.rewards  = torch.zeros(capacity, 1,          device=device)
        self.dones    = torch.zeros(capacity, 1,          device=device)

    def add(self, obs, next_obs, action, reward, done):
        n = obs.shape[0]
        idx = torch.arange(self.pos, self.pos + n, device=self.device) % self.cap
        self.obs[idx]      = obs
        self.next_obs[idx] = next_obs
        self.actions[idx]  = action
        self.rewards[idx]  = reward.unsqueeze(-1)
        self.dones[idx]    = done.float().unsqueeze(-1)
        self.pos = (self.pos + n) % self.cap
        self._size = min(self._size + n, self.cap)

    def sample(self, batch_size):
        idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        return (self.obs[idx], self.next_obs[idx], self.actions[idx],
                self.rewards[idx], self.dones[idx])

    def __len__(self):
        return self._size


# ── Training ───────────────────────────────────────────────────────────────────

def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Environment — rl-tools DEFAULT_PARAMETERS_FACTORY defaults
    env = QuadcopterEnv(
        num_envs=args.num_envs,
        config_path=args.config_path,
        max_episode_length_seconds=5.0,      # EPISODE_STEP_LIMIT=500 @ 100Hz
        rwd_scale=1.0,
        rwd_constant=0.5,                    # rl-tools: 0.5 (not 1.5)
        rwd_termination_penalty=-100.0,
        rwd_position=1.0,
        rwd_orientation=0.1,                 # rl-tools: 0.1 (not 0.2)
        rwd_d_action=1.0,
        term_position=1.0,                   # rl-tools: 1m (not 2m)
        term_linear_velocity=2.0,
        term_angular_velocity=35.0,
        init_guidance=0.1,
        init_max_position=0.5,               # rl-tools init_90_deg: 0.5m (not 1m)
        init_max_angle=np.pi / 2,
        init_max_linear_velocity=1.0,
        init_max_angular_velocity=1.0,
        dynamics_randomization_delta=args.dynamics_randomization_delta,
        device=args.device,
        use_compile=True,
    )

    obs_dim    = env.single_observation_space.shape[0]
    action_dim = env.single_action_space.shape[0]

    actor     = Actor(obs_dim, action_dim, args.hidden_size).to(device)
    qf1       = SoftQNetwork(obs_dim, action_dim, args.hidden_size).to(device)
    qf2       = SoftQNetwork(obs_dim, action_dim, args.hidden_size).to(device)
    qf1_target = SoftQNetwork(obs_dim, action_dim, args.hidden_size).to(device)
    qf2_target = SoftQNetwork(obs_dim, action_dim, args.hidden_size).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    q_optimizer     = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)

    # Auto-entropy tuning — matching rl-tools ENTROPY_BONUS=true
    target_entropy = -float(action_dim)
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha = log_alpha.exp().item()
    a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)

    rb = ReplayBuffer(args.buffer_size, obs_dim, action_dim, device)

    # Logging
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join("logs_sac", run_id)
    os.makedirs(log_dir, exist_ok=True)
    shutil.copy(args.config_path, os.path.join(log_dir, "flight_params.json"))

    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, group="sac", config=vars(args), name=f"sac_{run_id}")

    print(f"SAC | obs={obs_dim} act={action_dim} envs={args.num_envs} hidden={args.hidden_size}")
    print(f"Log: {log_dir}")

    obs, _ = env.reset()
    # drone_env returns tensors on device already
    if not isinstance(obs, torch.Tensor):
        obs = torch.as_tensor(obs, device=device, dtype=torch.float32)

    global_step = 0
    actor_loss  = torch.tensor(0.0)
    start_time  = time.time()

    try:
        while time.time() - start_time < args.train_minutes * 60:
            global_step += args.num_envs

            # Collect actions
            if global_step < args.learning_starts:
                actions = torch.zeros(args.num_envs, action_dim, device=device).uniform_(-1, 1)
            else:
                with torch.no_grad():
                    actions, _, _ = actor.get_action(obs)

            next_obs, rewards, terminated, truncated, infos = env.step(actions)

            if not isinstance(next_obs, torch.Tensor):
                next_obs = torch.as_tensor(next_obs, device=device, dtype=torch.float32)
            rewards_t   = rewards if isinstance(rewards, torch.Tensor) else torch.as_tensor(rewards, device=device, dtype=torch.float32)
            terminated_t = terminated if isinstance(terminated, torch.Tensor) else torch.as_tensor(terminated, device=device)

            # Store — use terminated only as done (not truncated) so we bootstrap
            # across time limits. next_obs for truncated envs is already the reset
            # obs (drone_env resets in-place), so bootstrapping is approximate.
            rb.add(obs, next_obs, actions.detach(), rewards_t, terminated_t)
            obs = next_obs

            # Training
            if global_step >= args.learning_starts and len(rb) >= args.batch_size:
                b_obs, b_next_obs, b_actions, b_rewards, b_dones = rb.sample(args.batch_size)

                # Critic update
                with torch.no_grad():
                    next_actions, next_log_pi, _ = actor.get_action(b_next_obs)
                    qf1_next = qf1_target(b_next_obs, next_actions)
                    qf2_next = qf2_target(b_next_obs, next_actions)
                    min_qf_next = torch.min(qf1_next, qf2_next) - alpha * next_log_pi
                    next_q = b_rewards + (1 - b_dones) * args.gamma * min_qf_next

                qf1_val  = qf1(b_obs, b_actions)
                qf2_val  = qf2(b_obs, b_actions)
                qf1_loss = F.mse_loss(qf1_val, next_q)
                qf2_loss = F.mse_loss(qf2_val, next_q)
                qf_loss  = qf1_loss + qf2_loss
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                # Actor update every 2 critic steps — rl-tools ACTOR_TRAINING_INTERVAL=2x
                if global_step % (args.policy_frequency * args.num_envs) == 0:
                    for _ in range(args.policy_frequency):
                        pi, log_pi, _ = actor.get_action(b_obs)
                        min_qf_pi = torch.min(qf1(b_obs, pi), qf2(b_obs, pi))
                        actor_loss = (alpha * log_pi - min_qf_pi).mean()
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                        with torch.no_grad():
                            _, log_pi2, _ = actor.get_action(b_obs)
                        alpha_loss = (-log_alpha.exp() * (log_pi2 + target_entropy)).mean()
                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

                # Soft target update
                if global_step % args.target_network_frequency == 0:
                    for p, tp in zip(qf1.parameters(), qf1_target.parameters()):
                        tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)
                    for p, tp in zip(qf2.parameters(), qf2_target.parameters()):
                        tp.data.copy_(args.tau * p.data + (1 - args.tau) * tp.data)

            # Logging
            if global_step % args.print_interval == 0:
                sps = int(global_step / (time.time() - start_time))
                ep_rew = infos[0].get("episode_reward_mean", float("nan")) if infos else float("nan")
                ep_len = infos[0].get("episode_length_mean", float("nan")) if infos else float("nan")
                print(f"step={global_step:>10,}  sps={sps:>6,}  ep_rew={ep_rew:>8.2f}  "
                      f"ep_len={ep_len:>6.0f}  alpha={alpha:.4f}  buf={len(rb):,}")
                if args.wandb:
                    import wandb
                    wandb.log(dict(
                        global_step=global_step, sps=sps,
                        episode_reward_mean=ep_rew, episode_length_mean=ep_len,
                        alpha=alpha, qf_loss=qf_loss.item(), actor_loss=actor_loss.item(),
                    ))

    finally:
        path = os.path.join(log_dir, "model.pt")
        torch.save({"actor": actor.state_dict(), "qf1": qf1.state_dict(),
                    "qf2": qf2.state_dict(), "global_step": global_step}, path)
        export_weights(actor, os.path.join(log_dir, "neural_network.c"), run_id=run_id)
        print(f"Saved to {path}")
        if args.wandb:
            import wandb
            wandb.finish()


def main():
    p = argparse.ArgumentParser()
    # Env
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--config-path", type=str, default="meteor75_parameters.json")
    p.add_argument("--dynamics-randomization-delta", type=float, default=0.0)
    # Network — rl-tools: hidden=32, tanh
    p.add_argument("--hidden-size", type=int, default=32)
    # SAC hyperparams — rl-tools defaults
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--buffer-size", type=int, default=10_000_000)
    p.add_argument("--learning-starts", type=int, default=1000)   # N_WARMUP_STEPS=1000
    p.add_argument("--policy-lr", type=float, default=1e-3)       # rl-tools: 1e-3
    p.add_argument("--q-lr", type=float, default=1e-3)            # rl-tools: 1e-3
    p.add_argument("--policy-frequency", type=int, default=2)     # actor every 2 critic steps
    p.add_argument("--target-network-frequency", type=int, default=1)
    # Run
    p.add_argument("--train-minutes", type=float, default=120.0)
    p.add_argument("--print-interval", type=int, default=10_000)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="puffer-raptor")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
