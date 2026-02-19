# puffer_raptor vs rl-tools L2F: Implementation Comparison

This document summarizes the theoretical and practical differences between the `puffer_raptor` quadcopter training implementation and the rl-tools `src/rl/environments/l2f/sac/` reference. It is intended as context for further discussion without requiring access to the full codebases.

---

## 1. Algorithm: PPO vs SAC

| | puffer_raptor | rl-tools |
|---|---|---|
| Algorithm | PPO (on-policy) | SAC (off-policy) |
| Data reuse | Each transition used once then discarded | Stored in replay buffer, reused indefinitely |
| Sample efficiency | Low — needs ~100M steps | High — converges in ~10M steps |
| Exploration | Entropy coefficient (manual tuning) | Automatic entropy tuning (α learned) |
| Parallelism | 8192 vectorized envs on GPU compensates for low sample efficiency | Single env; replay buffer makes parallelism less critical |

**Key insight**: SAC's replay buffer is the main reason rl-tools needs 10x fewer environment interactions. PPO throws away data after every update; SAC learns from every transition ever collected. SAC also can't collapse to low-entropy behavior the way PPO can (with `ent_coef=0`), because entropy is part of the objective by design.

---

## 2. Reward Function

Both use the same **rl-tools Squared reward** formula:

```
reward = -scale * weighted_cost + constant    (if not crashed)
       = termination_penalty                  (if crashed)

weighted_cost = position_weight   * ||pos - desired_pos||
              + orientation_weight * 2*acos(1 - |q_z|)
              + d_action_weight   * ||action - last_action||
```

Side-by-side parameter comparison:

| Parameter | rl-tools default | puffer_raptor current | Notes |
|---|---|---|---|
| `scale` | 1.0 | 1.0 | ✓ matches |
| `constant` | **0.5** | **1.5** | Significant difference — see below |
| `termination_penalty` | -100.0 | -100.0 | ✓ matches |
| `position` weight | 1.0 | 1.0 | ✓ matches |
| `orientation` weight | **0.1** | **0.2** | See cancellation note below |
| `linear_velocity` weight | 0.0 | 0.0 | ✓ matches |
| `angular_velocity` weight | 0.0 | 0.0 | ✓ matches |
| `action` weight | 0.0 | 0.0 | ✓ matches |
| `d_action` weight | 1.0 | 1.0 | ✓ matches |

### The `constant` difference

rl-tools uses `constant=0.5`, puffer_raptor uses `constant=1.5`. This is the most impactful difference:
- rl-tools: at perfect hover, reward = +0.5. With moderate error, reward drops toward 0.
- puffer_raptor: at perfect hover, reward = +1.5. Even with substantial error the reward stays positive.

The higher constant inflates rewards and reduces the gradient signal quality. The agent doesn't feel much difference between hovering well and hovering poorly because the constant dominates. **Recommended: lower to 0.5 to match rl-tools.**

### Orientation weight factor-of-2 cancellation

rl-tools computes orientation cost as `2 * acos(1 - |q_z|)` with weight `0.1`.
puffer_raptor computes `acos(1 - |q_z|)` (missing the factor of 2) with weight `0.2`.

These are numerically identical: `0.1 * 2 * acos(...)` = `0.2 * acos(...)`. The comment in `drone_env.py` says it matches rl-tools, which is true in effect but the factor-of-2 is absent in the cost and doubled in the weight. Not a bug functionally, but worth being aware of.

---

## 3. Termination Thresholds

| Condition | rl-tools | puffer_raptor |
|---|---|---|
| Position error (per axis) | **1.0 m** | **2.0 m** |
| Linear velocity (per axis) | 2.0 m/s | 2.0 m/s |
| Angular velocity (per axis) | 35.0 rad/s | 35.0 rad/s |

The tighter position threshold in rl-tools (1m vs 2m) means:
- Crashes happen sooner after divergence → clearer failure signal
- Episodes are shorter on average during early training → faster learning
- The agent is forced to stay closer to the goal to survive

**Recommended: tighten `term_position` to 1.0m.**

---

## 4. Episode Initialization

| Parameter | rl-tools (`init_90_deg`) | puffer_raptor |
|---|---|---|
| `guidance` probability | 0.1 | 0.1 |
| Max position offset | **0.5 m** | **1.0 m** |
| Max initial angle | π/2 (90°) | π/2 (90°) |
| Max linear velocity | 1.0 m/s | 1.0 m/s |
| Max angular velocity | 1.0 rad/s | 1.0 rad/s |
| Initial RPM | `[0, hover_rpm]` (relative) | **0 (rotors off)** |

Two meaningful differences:

1. **Position range**: puffer_raptor starts agents twice as far from the goal. Harder initial conditions slow early learning.

2. **Initial RPM**: rl-tools seeds rotors at a random speed between 0 and hover RPM. puffer_raptor starts rotors at zero, meaning the drone is in free-fall for the first ~90ms while motors spin up — a very different initial dynamic that the policy must learn to handle.

**Recommended: `init_max_position=0.5m`, and investigate seeding initial rotor speeds.**

---

## 5. Observation Space

| Component | rl-tools SAC (partial obs) | puffer_raptor |
|---|---|---|
| Position (3D) | ✓ | ✓ |
| Orientation | **IMU accelerometer (3D) + Magnetometer** | **Full 3×3 rotation matrix (9D)** |
| Linear velocity (3D) | ✓ | ✓ |
| Angular velocity (3D) | ✓ | ✓ |
| Rotor speeds | ✗ | ✓ (4D, scaled) |
| Action history | 1 step (4D) | 1 step (4D) |
| **Total dims** | ~16 | **26** |

rl-tools uses **partial observability** — no explicit rotation matrix. The agent must reconstruct orientation implicitly from IMU readings (linear acceleration in body frame) and magnetometer (body X-axis projected to global XY plane). rl-tools compensates by using a **GRU** (recurrent network) with a 10-step sequence length, so the network can integrate IMU over time.

puffer_raptor gives the agent the full rotation matrix directly — this is actually an *easier* observation problem (the agent has more information). However, it includes rotor speeds (4D) as an additional observation that rl-tools does not use in the default partial-obs configuration.

---

## 6. Network Architecture

| | rl-tools | puffer_raptor PPO | puffer_raptor SAC |
|---|---|---|---|
| Type | **GRU** (recurrent) | MLP | MLP |
| Hidden size | **32** | **64** | **32** (now matched) |
| Activation | **TANH** | **ELU** | **ELU** (PPO structure) |
| Sequence length | 10 steps | — | — |
| Log std | — | Global parameter | Global parameter |
| Value head | — | ✓ (PPO) | ✗ (SAC) |

The GRU in rl-tools is specifically to handle partial observability — the recurrent state accumulates orientation information from IMU history. Since puffer_raptor provides the full rotation matrix, a GRU is not needed.

The switch from TANH to ELU in puffer_raptor's SAC is a departure from rl-tools, made to keep the policy export-compatible with the existing C code generator (`export.py` emits `nn_elu`).

---

## 7. Training Hyperparameters

| Parameter | rl-tools SAC | puffer_raptor PPO | puffer_raptor SAC |
|---|---|---|---|
| Gamma (discount) | 0.99 | 0.999 | 0.99 |
| Actor LR | 1e-3 | 5e-4 (Muon) | 1e-3 |
| Critic/value LR | 1e-3 | 5e-4 (Muon) | 1e-3 |
| Batch size | 64 | 8192×128 (on-policy) | 64 |
| Warmup steps | 1000 | — | 1000 |
| Total steps | 10M | 100M | 10M target |
| Optimizer | Adam | **Muon** | Adam |
| Tau (soft update) | — | — | 0.005 |
| Actor update freq | every 20 env steps | every rollout | every 2 critic steps |

---

## 8. PPO-Specific Hyperparameters for Sparse vs Dense Rewards

This section captures the discussion about which PPO hyperparameters most affect learning with sparse/semi-sparse rewards.

### `gamma` (discount factor)
Currently 0.999 in puffer_raptor. At `bptt_horizon=128` steps and `dt=0.01s`, the effective lookahead is 1.28 seconds. Higher gamma extends the effective planning horizon. rl-tools uses 0.99 — a deliberate choice for their shorter 5-second episodes.

### `gae_lambda`
Controls bias-variance tradeoff in advantage estimation. Sparse rewards = use **higher lambda (0.97–0.99)** to reduce bias. Low lambda causes advantages to be dominated by the immediate TD error, washing out delayed reward signal.

### `ent_coef` (entropy coefficient)
Was 0.0 in PPO — the policy could collapse before finding reward signal. A small value (0.001–0.01) encourages exploration. SAC makes this moot by auto-tuning α.

### Rollout length (`bptt_horizon`)
In puffer_lib, `bptt_horizon` is both the rollout length and the GAE advantage window. With `batch_size = num_envs * 128` and `bptt_horizon = batch_size // num_envs = 128`, the advantage can only see 1.28 seconds ahead. Longer rollouts = better credit assignment for delayed rewards.

### `init_guidance`
Not a PPO hyperparameter, but the most impactful curriculum lever. With `init_guidance=0.1` only 10% of episodes start at the goal. A curriculum starting at 0.5–0.9 (agents near success) and decaying is often more impactful than any hyperparameter tuning for sparse rewards.

### `vf_coef`
Currently 2.0. High value loss coefficient makes sense for dense rewards but can destabilize with sparse rewards (value function has little signal to fit). Consider 0.5–1.0.

---

## Summary of Actionable Differences

In rough priority order of impact:

1. **`rwd_constant`: 1.5 → 0.5** — tighter reward signal, matches rl-tools
2. **`term_position`: 2.0 → 1.0 m** — faster failure signal, shorter bad episodes
3. **`init_max_position`: 1.0 → 0.5 m** — closer to rl-tools initial conditions
4. **Initial rotor speeds** — investigate seeding to [0, hover_rpm] instead of 0
5. **`ent_coef` for PPO: 0.0 → 0.005** — prevent policy collapse before finding reward
6. **`init_guidance` curriculum** — start at 0.5+, decay over training
7. **`gae_lambda`: raise toward 0.97–0.99** — reduce advantage bias for sparse signal
