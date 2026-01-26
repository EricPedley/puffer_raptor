from __future__ import annotations

import gymnasium as gym
import json
import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any

import gymnasium
import pufferlib

def quaternion_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions (w, x, y, z format)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack([w, x, y, z], dim=-1)


def quaternion_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Compute quaternion conjugate (w, x, y, z format)."""
    conj = q.clone()
    conj[..., 1:] *= -1
    return conj


def rotate_vector_by_quaternion(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a quaternion (w, x, y, z format)."""
    q_v = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    q_conj = quaternion_conjugate(q)
    rotated = quaternion_multiply(quaternion_multiply(q, q_v), q_conj)
    return rotated[..., 1:]


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to rotation matrix (w, x, y, z format)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = torch.zeros((*q.shape[:-1], 3, 3), device=q.device, dtype=q.dtype)
    R[..., 0, 0] = 1 - 2 * (y**2 + z**2)
    R[..., 0, 1] = 2 * (x*y - w*z)
    R[..., 0, 2] = 2 * (x*z + w*y)
    R[..., 1, 0] = 2 * (x*y + w*z)
    R[..., 1, 1] = 1 - 2 * (x**2 + z**2)
    R[..., 1, 2] = 2 * (y*z - w*x)
    R[..., 2, 0] = 2 * (x*z - w*y)
    R[..., 2, 1] = 2 * (y*z + w*x)
    R[..., 2, 2] = 1 - 2 * (x**2 + y**2)

    return R

class SamplePufferEnv(pufferlib.PufferEnv):
    def __init__(self, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(1,))
        self.single_action_space = gymnasium.spaces.Discrete(2)
        self.num_agents = 2
        super().__init__(buf)

    def reset(self, seed=0):
        self.observations[:] = self.observation_space.sample()
        return self.observations, []

    def step(self, action):
        self.observations[:] = self.observation_space.sample()
        infos = [{'infos': 'is a list of dictionaries'}]
        return self.observations, self.rewards, self.terminals, self.truncations, infos

if __name__ == '__main__':
    puffer_env = SamplePufferEnv()
    observations, infos = puffer_env.reset()
    actions = puffer_env.action_space.sample()
    observations, rewards, terminals, truncations, infos = puffer_env.step(actions)
    print('Puffer envs use a vector interface and in-place array updates')
    print('Observation:', observations)
    print('Reward:', rewards)
    print('Terminal:', terminals)
    print('Truncation:', truncations)


class QuadcopterEnv(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs: int = 1,
        config_path: str = "my_quad_parameters.json",
        max_episode_length: int = 500,
        dt: float = 0.01,
        lin_vel_reward_scale: float = -0.05,
        ang_vel_reward_scale: float = -0.01,
        distance_to_goal_reward_scale: float = 1.0,
        dynamics_randomization_delta: float = 0.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        render_mode: Optional[str] = None,
        **kwargs
    ):
        self.single_action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32
        )
        self.num_envs = num_envs
        self.num_agents = num_envs  # For PufferLib compatibility
        super().__init__()

        self.device = torch.device(device)
        self.dt = dt
        self.max_episode_length = max_episode_length
        self.max_episode_length_s = max_episode_length * dt
        self.lin_vel_reward_scale = lin_vel_reward_scale
        self.ang_vel_reward_scale = ang_vel_reward_scale
        self.distance_to_goal_reward_scale = distance_to_goal_reward_scale
        self.dynamics_randomization_delta = dynamics_randomization_delta
        self.render_mode = render_mode

        # Define action and observation spaces
        self.action_space = self.single_action_space
        self.observation_space = self.single_observation_space

        # Load quadcopter parameters
        params = json.load(open(config_path))

        # Quadcopter state
        self._position = torch.zeros(self.num_envs, 3, device=self.device)  # world frame
        self._velocity = torch.zeros(self.num_envs, 3, device=self.device)  # world frame
        self._quaternion = torch.zeros(self.num_envs, 4, device=self.device)  # (w, x, y, z)
        self._quaternion[:, 0] = 1.0  # identity quaternion
        self._angular_velocity = torch.zeros(self.num_envs, 3, device=self.device)  # body frame

        # Actions and forces
        self._actions = torch.zeros(self.num_envs, 4, device=self.device)
        self._rotor_speeds = torch.zeros(self.num_envs, 4, device=self.device)

        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Physics parameters
        self._mass = params['mass']
        self._inertia = torch.tensor(params['inertia_diag'], device=self.device)
        self._inertia_inv = 1.0 / self._inertia
        self._gravity = torch.tensor([0.0, 0.0, -9.81], device=self.device)

        # Episode tracking
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["lin_vel", "ang_vel", "distance_to_goal"]
        }

        # Store nominal (original) dynamics parameters
        self._nominal_thrust_coefficients = torch.tensor(params['thrust_coefficients'], device=self.device, dtype=torch.float32)
        self._nominal_thrust_directions = torch.tensor(params['rotor_thrust_directions'], dtype=torch.float32, device=self.device)
        self._nominal_rotor_torque_directions = torch.tensor(params['rotor_torque_directions'], dtype=torch.float32, device=self.device)
        self._nominal_rotor_torque_constants = torch.tensor(params['rotor_torque_constants'], dtype=torch.float32, device=self.device)
        self._nominal_rotor_positions = torch.tensor(params['rotor_positions'], dtype=torch.float32, device=self.device)
        self._nominal_rising_delay_constants = 1.0 / torch.tensor(params['delay_rising_constants'], dtype=torch.float32, device=self.device)
        self._nominal_falling_delay_constants = 1.0 / torch.tensor(params['delay_falling_constants'], dtype=torch.float32, device=self.device)

        # Create per-environment randomized dynamics parameters
        self._thrust_coefficients = self._nominal_thrust_coefficients.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self._thrust_directions = self._nominal_thrust_directions.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self._rotor_torque_directions = self._nominal_rotor_torque_directions.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self._rotor_torque_constants = self._nominal_rotor_torque_constants.unsqueeze(0).repeat(self.num_envs, 1)
        self._rotor_positions = self._nominal_rotor_positions.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self._rising_delay_constants = self._nominal_rising_delay_constants.unsqueeze(0).repeat(self.num_envs, 1)
        self._falling_delay_constants = self._nominal_falling_delay_constants.unsqueeze(0).repeat(self.num_envs, 1)

    def step(self, actions: torch.Tensor) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        # Ensure actions are on correct device and shape
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, device=self.device, dtype=torch.float32)

        # Ensure actions have batch dimension
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)  # (4,) -> (1, 4)

        # Process actions and apply physics
        self._actions = actions.clone().clamp(-1.0, 1.0)
        actions_0_1 = (self._actions + 1.0) / 2.0

        # Apply motor delay
        rising_mask = actions_0_1 > self._rotor_speeds
        falling_mask = actions_0_1 <= self._rotor_speeds
        diffs = actions_0_1 - self._rotor_speeds
        self._rotor_speeds[rising_mask] += (diffs * self._rising_delay_constants)[rising_mask] * self.dt
        self._rotor_speeds[falling_mask] += (diffs * self._falling_delay_constants)[falling_mask] * self.dt

        # Compute thrust from rotor speeds (quadratic thrust curve)
        actions_polynomial = torch.stack([
            torch.ones_like(self._rotor_speeds),
            self._rotor_speeds,
            torch.square(self._rotor_speeds)
        ], dim=-1)  # N x 4 x 3
        thrust_magnitude = torch.einsum('ijk,ijk->ij', actions_polynomial, self._thrust_coefficients)  # N x 4
        rotor_thrust = thrust_magnitude[..., torch.newaxis] * self._thrust_directions

        # Compute torques
        # Yaw moment (torque in z axis)
        torque_body = torch.sum(
            thrust_magnitude[..., torch.newaxis] *
            self._rotor_torque_constants[..., torch.newaxis] *
            self._rotor_torque_directions,
            dim=1
        )
        # Roll and pitch moment (torque in x and y axis)
        cross_prod = sum([
            torch.cross(self._rotor_positions[:, i, :], rotor_thrust[:, i, :], dim=-1)
            for i in range(4)
        ])
        torque_body += cross_prod

        # Total thrust in body frame
        total_thrust_body = rotor_thrust.sum(dim=1)

        # Integrate physics
        self._integrate_physics(total_thrust_body, torque_body)

        # Get observations
        self.observations = self._get_observations()

        # Compute rewards
        self.rewards, rewards_dict = self._get_rewards()

        # Check for termination
        self.terminals, self.truncations = self._get_dones()

        # Update episode length
        self.episode_length_buf += 1

        # Handle resets
        reset_envs = torch.where(self.terminals | self.truncations)[0]
        if len(reset_envs) > 0:
            self._reset_idx(reset_envs)

        # Compute reward statistics across all environments
        info = {
            "mean_reward": self.rewards.mean().item(),
        }

        # Add mean for each reward component
        for key, value in rewards_dict.items():
            info[f"mean_{key}"] = value.mean().item()

        self.infos = [info]
        return (self.observations, self.rewards, self.terminals,
            self.truncations, self.infos)

    def _integrate_physics(self, thrust_body: torch.Tensor, torque_body: torch.Tensor):
        """Integrate quadcopter dynamics using Euler integration."""
        # Convert thrust from body to world frame
        thrust_world = rotate_vector_by_quaternion(thrust_body, self._quaternion)

        # Linear acceleration (F/m + g)
        linear_acc = thrust_world / self._mass + self._gravity

        # Update velocity and position
        self._velocity += linear_acc * self.dt
        self._position += self._velocity * self.dt

        # Angular acceleration (I^-1 * (tau - omega x (I * omega)))
        I_omega = self._inertia * self._angular_velocity
        gyroscopic = torch.cross(self._angular_velocity, I_omega, dim=-1)
        angular_acc = self._inertia_inv * (torque_body - gyroscopic)
        assert not torch.isinf(angular_acc).any().item()

        # Update angular velocity
        self._angular_velocity = torch.clamp(self._angular_velocity + angular_acc * self.dt, -1e9, 1e9)
        assert not torch.isinf(self._angular_velocity).any().item()

        # Update quaternion
        # dq/dt = 0.5 * q * omega_quat
        omega_quat = torch.cat([
            torch.zeros_like(self._angular_velocity[..., :1]),
            self._angular_velocity
        ], dim=-1)
        q_dot = 0.5 * quaternion_multiply(self._quaternion, omega_quat)
        assert not torch.isnan(q_dot).any().item()
        self._quaternion += q_dot * self.dt

        # Normalize quaternion
        self._quaternion = self._quaternion / torch.norm(self._quaternion, dim=-1, keepdim=True)
        assert not torch.isnan(self._quaternion).any().item()

    def _get_observations(self) -> np.ndarray:
        """Compute observations for all environments."""
        # Get rotation matrix
        R = quaternion_to_rotation_matrix(self._quaternion)

        # Transform velocity to body frame
        velocity_body = torch.einsum('bij,bj->bi', R.transpose(-2, -1), self._velocity)

        # Project gravity to body frame
        gravity_body = torch.einsum('bij,bj->bi', R.transpose(-2, -1), self._gravity.unsqueeze(0).expand(self.num_envs, -1))

        # Transform desired position to body frame (relative position)
        rel_pos_world = self._desired_pos_w - self._position
        rel_pos_body = torch.einsum('bij,bj->bi', R.transpose(-2, -1), rel_pos_world)

        obs = torch.cat([
            velocity_body,           # 3
            self._angular_velocity,  # 3
            gravity_body,            # 3
            rel_pos_body,            # 3
        ], dim=-1)


        return obs

    def _get_rewards(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute rewards for all environments.

        Returns:
            Tuple of (total_reward, rewards_dict) where rewards_dict contains
            individual reward components for each environment.
        """
        # Get velocity in body frame for reward calculation
        R = quaternion_to_rotation_matrix(self._quaternion)
        velocity_body = torch.einsum('bij,bj->bi', R.transpose(-2, -1), self._velocity)

        lin_vel = torch.sum(torch.square(velocity_body), dim=1)
        ang_vel = torch.sum(torch.square(self._angular_velocity), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._position, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)

        rewards = {
            "lin_vel": lin_vel * self.lin_vel_reward_scale * self.dt,
            "ang_vel": ang_vel * self.ang_vel_reward_scale * self.dt,
            "distance_to_goal": distance_to_goal_mapped * self.distance_to_goal_reward_scale * self.dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)

        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value

        return reward, rewards

    def _get_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check for episode termination."""
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._position[:, 2] < 0.1, self._position[:, 2] > 2.0)
        return died, time_out

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset all environments."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # Reset all environments
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(env_ids)

        obs = self._get_observations()  # Already flattened
        return obs, [dict()]

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specific environments."""
        if len(env_ids) == 0:
            return

        # Reset episode tracking
        self.episode_length_buf[env_ids] = 0
        self._actions[env_ids] = 0.0
        self._rotor_speeds[env_ids] = 0.0

        # Reset episode sums
        for key in self._episode_sums.keys():
            self._episode_sums[key][env_ids] = 0.0

        # Randomize dynamics parameters for reset environments
        delta = self.dynamics_randomization_delta
        num_reset = len(env_ids)

        if delta > 0:
            # Generate random multipliers: (1 +- delta)
            self._thrust_coefficients[env_ids] = self._nominal_thrust_coefficients * (
                1.0 + torch.zeros((num_reset, *self._nominal_thrust_coefficients.shape), device=self.device).uniform_(-delta, delta)
            )
            self._rotor_torque_constants[env_ids] = self._nominal_rotor_torque_constants * (
                1.0 + torch.zeros((num_reset, *self._nominal_rotor_torque_constants.shape), device=self.device).uniform_(-delta, delta)
            )
            self._rotor_positions[env_ids] = self._nominal_rotor_positions * (
                1.0 + torch.zeros((num_reset, *self._nominal_rotor_positions.shape), device=self.device).uniform_(-delta, delta)
            )
            self._rising_delay_constants[env_ids] = self._nominal_rising_delay_constants * (
                1.0 + torch.zeros((num_reset, *self._nominal_rising_delay_constants.shape), device=self.device).uniform_(-delta, delta)
            )
            self._falling_delay_constants[env_ids] = self._nominal_falling_delay_constants * (
                1.0 + torch.zeros((num_reset, *self._nominal_falling_delay_constants.shape), device=self.device).uniform_(-delta, delta)
            )

        # Sample new goal positions
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

        # Reset quadcopter state to origin with identity orientation
        self._position[env_ids] = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self._velocity[env_ids] = 0.0
        self._quaternion[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self._angular_velocity[env_ids] = 0.0

    def close(self):
        pass