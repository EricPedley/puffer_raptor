from __future__ import annotations

import gymnasium as gym
import json
import torch
import numpy as np
from typing import Optional, Tuple, Dict, Any

import gymnasium
import pufferlib

try:
    import rerun as rr
    from logging_utils import log_drone_pose
    HAS_RERUN = True
except ImportError:
    HAS_RERUN = False

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
    """
    Rotate a vector by a quaternion (w, x, y, z format).
    v is in body frame and q is the transform from body to world frame.
    The result is v in world frame.
    """
    q_v = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    q_conj = quaternion_conjugate(q)
    rotated = quaternion_multiply(quaternion_multiply(q, q_v), q_conj)
    return rotated[..., 1:]

def rotate_vector_by_quaternion_conj(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Rotate a vector by a quaternion (w, x, y, z format).
    v is in world frame and q is the transform from body to world frame.
    The result is v in body frame.
    """
    q_v = torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1)
    q_conj = quaternion_conjugate(q)
    rotated = quaternion_multiply(quaternion_multiply(q_conj, q_v), q)
    return rotated[..., 1:]


def quaternion_error_axis_angle(q_current: torch.Tensor, q_desired: torch.Tensor) -> torch.Tensor:
    """
    Compute the axis-angle error between current and desired quaternions.
    Returns the axis-angle representation (3D vector where magnitude is angle).
    Result is in the body frame of q_current.

    q_current, q_desired: quaternions in (w, x, y, z) format
    """
    # q_error = q_desired * q_current^-1 gives rotation from current to desired
    # But we want it in body frame, so: q_error = q_current^-1 * q_desired
    q_current_inv = quaternion_conjugate(q_current)
    q_error = quaternion_multiply(q_current_inv, q_desired)

    # Ensure we take the shortest path (q and -q represent same rotation)
    # If w < 0, negate the quaternion
    sign = torch.sign(q_error[..., 0:1])
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q_error = q_error * sign

    # Extract axis-angle from quaternion
    # q = [cos(theta/2), sin(theta/2) * axis]
    w = q_error[..., 0:1]
    xyz = q_error[..., 1:]

    # sin(theta/2) = ||xyz||
    sin_half_angle = torch.norm(xyz, dim=-1, keepdim=True)

    # Handle small angles to avoid division by zero
    # For small angles: theta ≈ 2 * sin(theta/2), axis ≈ xyz / sin(theta/2)
    # So axis_angle ≈ 2 * xyz
    small_angle_mask = sin_half_angle < 1e-6

    # For larger angles: theta = 2 * atan2(sin(theta/2), cos(theta/2))
    half_angle = torch.atan2(sin_half_angle, w)
    angle = 2.0 * half_angle

    # axis = xyz / sin(theta/2), axis_angle = angle * axis
    axis_angle = torch.where(
        small_angle_mask,
        2.0 * xyz,  # Small angle approximation
        angle * xyz / (sin_half_angle + 1e-10)  # Normal case
    )

    return axis_angle


def quaternion_from_z_rotation(yaw: torch.Tensor) -> torch.Tensor:
    """
    Create quaternion from yaw angle (rotation about z-axis).
    yaw: tensor of yaw angles in radians
    Returns: quaternion in (w, x, y, z) format
    """
    half_yaw = yaw / 2.0
    w = torch.cos(half_yaw)
    x = torch.zeros_like(yaw)
    y = torch.zeros_like(yaw)
    z = torch.sin(half_yaw)
    return torch.stack([w, x, y, z], dim=-1)


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

class QuadcopterEnv(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs: int = 1,
        config_path: str = "my_quad_parameters.json",
        max_episode_length: int = 1000,
        dt: float = 0.01,
        lin_vel_reward_scale: float = -0.05,
        ang_vel_reward_scale: float = -0.01,
        distance_to_goal_reward_scale: float = 15.0,
        orientation_reward_scale: float = 5.0,
        dynamics_randomization_delta: float = 0.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        render_mode: Optional[str] = None,
        **kwargs
    ):
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32) # should be (-1, 1) but in isaaclab it's inf so we're sticking with that
        # Observations: velocity_body (3) + angular_velocity (3) + gravity_body (3) +
        #               rel_pos_body (3) + orientation_error_axis_angle (3) + rpm_scaled (4) = 19
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32
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
        self.orientation_reward_scale = orientation_reward_scale
        self.dynamics_randomization_delta = dynamics_randomization_delta
        self.render_mode = render_mode

        # Initialize rerun logging if rendering in human mode
        if self.render_mode == "human" and HAS_RERUN:
            rr.init("quadcopter_env", spawn=True)

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

        # Goal orientation (quaternion, z-axis rotations only)
        self._desired_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._desired_quat_w[:, 0] = 1.0  # identity quaternion

        # Physics parameters
        self._mass = params['mass']
        self._inertia = torch.tensor(params['inertia_diag'], device=self.device)
        self._inertia_inv = 1.0 / self._inertia
        self._gravity = torch.tensor([0.0, 0.0, -9.81], device=self.device)
        self._gravity_unit = torch.tensor([0.0, 0.0, -1.0], device=self.device)

        self._max_rpm = params['max_measured_rpm']

        # Episode tracking
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in ["lin_vel", "ang_vel", "distance_to_goal", "orientation"]
        }
        self._cumulative_rewards = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Completed episode statistics (stores most recent completed episode for each env)
        self._completed_episode_lengths = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self._completed_episode_rewards = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Store nominal (original) dynamics parameters
        self._nominal_thrust_coefficients = torch.tensor(params['thrust_coefficients'], device=self.device, dtype=torch.float32)
        self._nominal_thrust_coefficients[:, 1] /= self._max_rpm
        self._nominal_thrust_coefficients[:, 2] /= self._max_rpm**2
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
        self._decimation_steps = 2

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
        actions_0_1 = self._max_rpm * (self._actions + 1.0) / 2.0
        for _ in range(self._decimation_steps-1):
            self._step_once(actions_0_1)
        return self._step_once(actions_0_1)
    
    def _step_once(self, actions_0_1: torch.Tensor):
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
            torch.cross(self._rotor_positions[:, i, :], rotor_thrust[:, i, :])#, dim=-1)
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

        # Accumulate rewards for episode tracking
        self._cumulative_rewards += self.rewards

        # Check for termination
        self.terminals, self.truncations = self._get_dones()

        # Update episode length
        self.episode_length_buf += 1

        # Handle resets
        reset_envs = torch.where(self.terminals | self.truncations)[0]
        if len(reset_envs) > 0:
            # Store completed episode stats before resetting
            self._completed_episode_lengths[reset_envs] = self.episode_length_buf[reset_envs].float()
            self._completed_episode_rewards[reset_envs] = self._cumulative_rewards[reset_envs]
            self._reset_idx(reset_envs)

        # Render if human mode is enabled
        if self.render_mode == "human" and HAS_RERUN:
            self._render()

        # Compute reward statistics across all environments
        info = {
            "mean_reward": self.rewards.mean().item(),
        }

        # Add mean for each reward component
        for key, value in rewards_dict.items():
            info[f"mean_{key}"] = value.mean().item()

        # Add episode statistics (min/max/mean across most recent completed episode per env)
        info["episode_length_min"] = self._completed_episode_lengths.min().item()
        info["episode_length_max"] = self._completed_episode_lengths.max().item()
        info["episode_length_mean"] = self._completed_episode_lengths.mean().item()
        info["episode_reward_min"] = self._completed_episode_rewards.min().item()
        info["episode_reward_max"] = self._completed_episode_rewards.max().item()
        info["episode_reward_mean"] = self._completed_episode_rewards.mean().item()

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

        # rl_tools::utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J, state.angular_velocity, vector);
        # // flops: 6
        # rl_tools::utils::vector_operations::cross_product<DEVICE, T>(state.angular_velocity, vector, vector2);
        # rl_tools::utils::vector_operations::sub<DEVICE, T, 3>(torque, vector2, vector);
        # // flops: 9
        # rl_tools::utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J_inv, vector, state_change.angular_velocity);

        # Update angular velocity
        self._angular_velocity = torch.clamp(self._angular_velocity + angular_acc * self.dt, -1e12, 1e12)
        # idk why angular velocity gets this high but when it does, it crashes training because it causes NaNs down the line. The clamp fixes this.

        # Update quaternion
        # dq/dt = 0.5 * q * omega_quat
        omega_quat = torch.cat([
            torch.zeros_like(self._angular_velocity[..., :1]),
            self._angular_velocity
        ], dim=-1)
        q_dot = 0.5 * quaternion_multiply(self._quaternion, omega_quat)
        self._quaternion += q_dot * self.dt

        # Normalize quaternion
        self._quaternion = self._quaternion / torch.norm(self._quaternion, dim=-1, keepdim=True)

    def _get_observations(self) -> np.ndarray:
        """Compute observations for all environments."""
        # Get rotation matrix
        # R = quaternion_to_rotation_matrix(self._quaternion)

        # Transform velocity to body frame

        velocity_body = rotate_vector_by_quaternion_conj(self._velocity, self._quaternion)

        # Project gravity to body frame
        gravity_body = rotate_vector_by_quaternion_conj(self._gravity_unit.unsqueeze(0).expand(self.num_envs, -1), self._quaternion)

        # Transform desired position to body frame (relative position)
        rel_pos_world = self._desired_pos_w - self._position
        rel_pos_body = rotate_vector_by_quaternion_conj(rel_pos_world, self._quaternion)

        rpm_scaled = self._rotor_speeds / self._max_rpm

        # Compute orientation error as axis-angle (in body frame)
        orientation_error = quaternion_error_axis_angle(self._quaternion, self._desired_quat_w)

        obs = torch.cat([
            velocity_body,           # 3
            self._angular_velocity,  # 3
            gravity_body,            # 3
            rel_pos_body,            # 3
            orientation_error,       # 3
            rpm_scaled               # 4
        ], dim=-1)

        assert not torch.isnan(obs).any()
        assert not torch.isinf(obs).any()


        return obs

    def _get_rewards(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute rewards for all environments.

        Returns:
            Tuple of (total_reward, rewards_dict) where rewards_dict contains
            individual reward components for each environment.
        """
        # Get velocity in body frame for reward calculation
        # R = quaternion_to_rotation_matrix(self._quaternion)
        # velocity_body = torch.einsum('bij,bj->bi', R.transpose(-2, -1), self._velocity)
        velocity_body = rotate_vector_by_quaternion_conj(self._velocity, self._quaternion)

        lin_vel = torch.sum(torch.square(velocity_body), dim=1)
        ang_vel = torch.sum(torch.square(self._angular_velocity), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._position, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)

        # Orientation error reward (using axis-angle magnitude)
        orientation_error = quaternion_error_axis_angle(self._quaternion, self._desired_quat_w)
        orientation_error_magnitude = torch.linalg.norm(orientation_error, dim=1)
        orientation_reward_mapped = 1 - torch.tanh(orientation_error_magnitude / 0.5)

        rewards = {
            "lin_vel": lin_vel * self.dt,
            "ang_vel": ang_vel * self.dt,
            "distance_to_goal": distance_to_goal_mapped * self.dt,
            "orientation": orientation_reward_mapped * self.dt, # scale orientation reward by distance to goal
        }
        scaled_rewards = {
            "lin_vel": lin_vel * self.lin_vel_reward_scale * self.dt,
            "ang_vel": ang_vel * self.ang_vel_reward_scale * self.dt,
            "distance_to_goal": distance_to_goal_mapped * self.distance_to_goal_reward_scale * self.dt,
            "orientation": orientation_reward_mapped * self.orientation_reward_scale * self.dt, # scale orientation reward by distance to goal
        }
        reward = torch.sum(torch.stack(list(scaled_rewards.values())), dim=0)

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
        self._cumulative_rewards[env_ids] = 0.0

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

        # Sample new goal orientations (z-axis rotations only)
        yaw_angles = torch.zeros(len(env_ids), device=self.device).uniform_(-np.pi, np.pi)
        self._desired_quat_w[env_ids] = quaternion_from_z_rotation(yaw_angles)

        # Reset quadcopter state to origin with identity orientation
        self._position[env_ids] = torch.tensor([0.0, 0.0, 1.0], device=self.device)
        self._velocity[env_ids] = 0.0
        self._quaternion[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self._angular_velocity[env_ids] = 0.0

    def _render(self):
        """Render the environment using rerun logging."""
        if not HAS_RERUN:
            return

        # Log the first environment's state (index 0)
        position = self._position[0].detach().cpu().numpy()
        quaternion = self._quaternion[0].detach().cpu().numpy()
        quat_xyzw = np.array([quaternion[1], quaternion[2],
                              quaternion[3], quaternion[0]])

        for i, action_val in enumerate(self._actions[0].cpu().numpy()):
            rr.log(f"actions/motor_{i}", rr.Scalars(float(action_val)))

        for i, observation_val in enumerate(self.observations[0].cpu().numpy()):
            rr.log(f"observations/{i}", rr.Scalars(float(observation_val)))

        log_drone_pose(position, quat_xyzw)

        # Log goal position with goal orientation
        goal_position = self._desired_pos_w[0].detach().cpu().numpy()
        goal_quaternion = self._desired_quat_w[0].detach().cpu().numpy()
        goal_quat_xyzw = np.array([goal_quaternion[1], goal_quaternion[2],
                                   goal_quaternion[3], goal_quaternion[0]])
        rr.log(
            "goal",
            rr.Transform3D(
                translation=goal_position,
                quaternion=goal_quat_xyzw,
            ),
            rr.TransformAxes3D(0.5),
            static=False,
        )

    def close(self):
        pass