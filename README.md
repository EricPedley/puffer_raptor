# Puffer Raptor - Quadcopter RL Training

A standalone quadcopter reinforcement learning environment with PufferLib PPO training, independent of Isaac Lab. The repo name comes from the paper that provided the inspiration for this project and the dynamics model: https://raptor.rl.tools/

## Overview

This project provides:
- A custom quadcopter physics simulation environment (`drone_env.py`)
- PPO training script using PufferLib (`train_ppo.py`)
- Vectorized environments for efficient parallel training

## Installation

```bash
# Install dependencies using uv
uv sync
```

## Environment Details

The `QuadcopterEnv` is a vectorized Gymnasium environment that simulates quadcopter dynamics:

- **Observation space**: 12-dimensional vector
  - Linear velocity (body frame): 3D
  - Angular velocity (body frame): 3D
  - Projected gravity (body frame): 3D
  - Relative goal position (body frame): 3D

- **Action space**: 4-dimensional continuous actions (rotor commands in [-1, 1])

- **Physics**: Custom rigid body dynamics with:
  - Quadratic thrust curves
  - Motor delays
  - Gyroscopic effects
  - Configurable dynamics randomization

## Usage

### Test the Environment

```bash
uv run test_env.py
```

### Train a PPO Agent

Basic training:
```bash
uv run train_ppo.py
```

With custom parameters:
```bash
uv run train_ppo.py \
  --num-envs 1024 \
  --total-timesteps 50000000 \
  --learning-rate 3e-4 \
  --batch-size 8192 \
  --dynamics-randomization-delta 0.2
```

Key training parameters:
- `--num-envs`: Number of parallel environments (default: 512)
- `--total-timesteps`: Total training steps (default: 10M)
- `--learning-rate`: Learning rate (default: 3e-4)
- `--batch-size`: Batch size for training (default: 4096)
- `--dynamics-randomization-delta`: Randomization range for physics parameters (default: 0.1)
- `--wandb`: Enable Weights & Biases logging

### Configuration

Quadcopter parameters are defined in `my_quad_parameters.json`:
- Mass and inertia
- Rotor positions and thrust characteristics
- Motor delay constants
- Torque coefficients

## Training Tips

1. **Parallelization**: Use more environments (`--num-envs`) for faster learning
2. **Dynamics Randomization**: Increase `--dynamics-randomization-delta` for more robust policies
3. **Batch Size**: Match or exceed `num_envs * episode_length` for efficient sampling
4. **Checkpointing**: Set `--checkpoint-interval` to save periodic checkpoints

## Key Changes from Isaac Lab Version

The environment has been modified to:
- Remove all Isaac Lab/Isaac Sim dependencies
- Implement custom rigid body physics simulation
- Use pure PyTorch for GPU-accelerated parallel simulation
- Provide a standard Gymnasium interface
- Support PufferLib's vectorized training

## File Structure

- `drone_env.py` - Standalone quadcopter environment
- `train_ppo.py` - PufferLib PPO training script
- `test_env.py` - Environment testing script
- `my_quad_parameters.json` - Quadcopter configuration
- `pyproject.toml` - Python dependencies

## Performance

The vectorized environment can simulate hundreds or thousands of quadcopters in parallel on GPU, enabling very fast training (millions of steps per second on modern GPUs).
