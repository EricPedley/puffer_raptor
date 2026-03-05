from __future__ import annotations

import json
import gymnasium as gym
import numpy as np
import torch
import rerun as rr
from typing import Optional, Tuple, Dict

import pufferlib

from quaternion_utils import (
    quaternion_multiply,
    rotate_vector_by_quaternion,
)
from logging_utils import log_drone_pose, log_gates


class QuadcopterRaceEnv(pufferlib.PufferEnv):
    """
    Gate-racing environment using drone_env.py's physics model and
    quad_race_env.py's observation/reward structure.

    Coordinate convention: z-up (positive z = up, gravity = [0, 0, -9.81]).
    Gate positions and start_position must be provided in z-up world frame.

    Observation (16 + 4*gates_ahead dims):
      [0:3]   pos_gate      - position relative to target gate in gate frame
      [3:6]   vel_gate      - velocity in gate frame
      [6:9]   euler_gate    - roll (phi), pitch (theta), yaw relative to gate
      [9:12]  ang_vel       - body-frame angular rates
      [12:16] rpm_scaled    - rotor speeds normalized to [-1, 1]
      [16:]   future_gates  - for each upcoming gate: (rel_x, rel_y, rel_z, rel_yaw)
                              in the current gate's reference frame
    """

    def __init__(
        self,
        gate_positions: np.ndarray,
        gate_yaws: np.ndarray,
        start_position: np.ndarray,
        num_envs: int = 1,
        config_path: str = "my_quad_parameters.json",
        gates_ahead: int = 1,
        max_episode_length: int = 5000,
        dt: float = 0.01,
        gate_hole_size: float = 1.5,
        gate_outer_size: float = 2.7,
        dynamics_randomization_delta: float = 0.0,
        initialize_at_random_gates: bool = True,
        autoreset: bool = True,
        render_mode: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        **kwargs,
    ):
        obs_dim = 16 + 4 * gates_ahead
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.single_action_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.num_envs = num_envs
        self.num_agents = num_envs
        super().__init__()

        self.device = torch.device(device)
        self.dt = dt
        self.max_episode_length = max_episode_length
        self.gates_ahead = gates_ahead
        self.gate_hole_size = gate_hole_size
        self.gate_outer_size = gate_outer_size
        self.dynamics_randomization_delta = dynamics_randomization_delta
        self.initialize_at_random_gates = initialize_at_random_gates
        self.autoreset = autoreset
        self.render_mode = render_mode

        self.num_gates = gate_positions.shape[0]
        self._gate_positions = torch.tensor(gate_positions, dtype=torch.float32, device=self.device)
        self._gate_yaws = torch.tensor(gate_yaws, dtype=torch.float32, device=self.device)
        self._start_position = torch.tensor(start_position, dtype=torch.float32, device=self.device)


        # Precompute relative gate info (position/yaw of gate i in gate i-1's frame)
        self._gate_pos_rel = torch.zeros(self.num_gates, 3, dtype=torch.float32, device=self.device)
        self._gate_yaw_rel = torch.zeros(self.num_gates, dtype=torch.float32, device=self.device)
        self._precompute_relative_gates()

        # Load physics parameters from JSON
        params = json.load(open(config_path))
        self._mass = params['mass']
        self._inertia = torch.tensor(params['inertia_diag'], dtype=torch.float32, device=self.device)
        self._inertia_inv = 1.0 / self._inertia
        self._gravity = torch.tensor([0.0, 0.0, -9.81], device=self.device)
        self._max_rpm = params['max_measured_rpm']

        # Nominal dynamics tensors
        self._nominal_thrust_coefficients = torch.tensor(params['thrust_coefficients'], dtype=torch.float32, device=self.device)
        self._nominal_thrust_directions = torch.tensor(params['rotor_thrust_directions'], dtype=torch.float32, device=self.device)
        self._nominal_rotor_torque_directions = torch.tensor(params['rotor_torque_directions'], dtype=torch.float32, device=self.device)
        self._nominal_rotor_torque_constants = torch.tensor(params['rotor_torque_constants'], dtype=torch.float32, device=self.device)
        self._nominal_rotor_positions = torch.tensor(params['rotor_positions'], dtype=torch.float32, device=self.device)
        self._nominal_rising_delay_constants = 1.0 / torch.tensor(params['delay_rising_constants'], dtype=torch.float32, device=self.device)
        self._nominal_falling_delay_constants = 1.0 / torch.tensor(params['delay_falling_constants'], dtype=torch.float32, device=self.device)

        # Per-env randomized dynamics
        self._thrust_coefficients = self._nominal_thrust_coefficients.unsqueeze(0).repeat(num_envs, 1, 1)
        self._thrust_directions = self._nominal_thrust_directions.unsqueeze(0).repeat(num_envs, 1, 1)
        self._rotor_torque_directions = self._nominal_rotor_torque_directions.unsqueeze(0).repeat(num_envs, 1, 1)
        self._rotor_torque_constants = self._nominal_rotor_torque_constants.unsqueeze(0).repeat(num_envs, 1)
        self._rotor_positions = self._nominal_rotor_positions.unsqueeze(0).repeat(num_envs, 1, 1)
        self._rising_delay_constants = self._nominal_rising_delay_constants.unsqueeze(0).repeat(num_envs, 1)
        self._falling_delay_constants = self._nominal_falling_delay_constants.unsqueeze(0).repeat(num_envs, 1)

        # Quadcopter state
        self._position = torch.zeros(num_envs, 3, device=self.device)
        self._velocity = torch.zeros(num_envs, 3, device=self.device)
        self._quaternion = torch.zeros(num_envs, 4, device=self.device)
        self._quaternion[:, 0] = 1.0  # identity
        self._angular_velocity = torch.zeros(num_envs, 3, device=self.device)
        self._rotor_speeds = torch.zeros(num_envs, 4, device=self.device)
        self._actions = torch.zeros(num_envs, 4, device=self.device)

        # Race-specific state
        self._target_gate_idx = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._prev_dist_to_gate = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

        # Episode tracking
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._cumulative_rewards = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self._completed_episode_lengths = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self._completed_episode_rewards = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

        # Initialize rerun if rendering
        self._gates_logged = False
        if self.render_mode == "human":
            rr.init("quadcopter_race_env", spawn=True)

        # torch.compile the physics kernel with all dynamics tensors as explicit args
        self._compiled_physics_step = torch.compile(
            self._physics_step_impl,
            mode="max-autotune",
            fullgraph=False,
        )
        # Warmup to trigger JIT compilation before training begins
        _w = torch.zeros(self.num_envs, 4, device=self.device)
        _wq = torch.zeros(self.num_envs, 4, device=self.device); _wq[:, 0] = 1.0
        self._compiled_physics_step(
            _w, _w,
            torch.zeros(self.num_envs, 3, device=self.device),
            torch.zeros(self.num_envs, 3, device=self.device),
            _wq,
            torch.zeros(self.num_envs, 3, device=self.device),
            self._rotor_positions, self._thrust_directions, self._thrust_coefficients,
            self._rotor_torque_constants, self._rotor_torque_directions,
            self._rising_delay_constants, self._falling_delay_constants,
            self._mass, self._inertia, self._inertia_inv, self._gravity, self.dt,
        )

    def _precompute_relative_gates(self):
        """Compute position/yaw of gate i in gate (i-1)'s reference frame."""
        for i in range(self.num_gates):
            prev = (i - 1) % self.num_gates
            delta = self._gate_positions[i] - self._gate_positions[prev]
            cos_y = torch.cos(self._gate_yaws[prev])
            sin_y = torch.sin(self._gate_yaws[prev])
            # Rotate xy delta into previous gate's frame
            self._gate_pos_rel[i, 0] = cos_y * delta[0] + sin_y * delta[1]
            self._gate_pos_rel[i, 1] = -sin_y * delta[0] + cos_y * delta[1]
            self._gate_pos_rel[i, 2] = delta[2]
            # Relative yaw wrapped to [-pi, pi]
            rel_yaw = self._gate_yaws[i] - self._gate_yaws[prev]
            rel_yaw = (rel_yaw + torch.pi) % (2 * torch.pi) - torch.pi
            self._gate_yaw_rel[i] = rel_yaw

    def _physics_step_impl(
        self,
        actions_0_1: torch.Tensor,
        rotor_speeds: torch.Tensor,
        position: torch.Tensor,
        velocity: torch.Tensor,
        quaternion: torch.Tensor,
        angular_velocity: torch.Tensor,
        rotor_positions: torch.Tensor,
        thrust_directions: torch.Tensor,
        thrust_coefficients: torch.Tensor,
        rotor_torque_constants: torch.Tensor,
        rotor_torque_directions: torch.Tensor,
        rising_delay_constants: torch.Tensor,
        falling_delay_constants: torch.Tensor,
        mass: float,
        inertia: torch.Tensor,
        inertia_inv: torch.Tensor,
        gravity: torch.Tensor,
        dt: float,
    ):
        """Pure physics kernel — all dynamics tensors passed explicitly for torch.compile."""
        # Motor delay
        rising_mask = actions_0_1 > rotor_speeds
        diffs = actions_0_1 - rotor_speeds
        delay_constants = torch.where(rising_mask, rising_delay_constants, falling_delay_constants)
        new_rotor_speeds = rotor_speeds + diffs * delay_constants * dt

        # Thrust (quadratic curve)
        actions_poly = torch.stack([
            torch.ones_like(new_rotor_speeds),
            new_rotor_speeds,
            torch.square(new_rotor_speeds),
        ], dim=-1)  # (N, 4, 3)
        thrust_magnitude = torch.einsum('ijk,ijk->ij', actions_poly, thrust_coefficients)  # (N, 4)
        rotor_thrust = thrust_magnitude[..., None] * thrust_directions  # (N, 4, 3)

        # Torques
        torque_body = torch.sum(
            thrust_magnitude[..., None] * rotor_torque_constants[..., None] * rotor_torque_directions,
            dim=1,
        )
        cross_prod = torch.cross(rotor_positions, rotor_thrust, dim=-1).sum(dim=1)
        torque_body = torque_body + cross_prod

        # Total thrust in body frame
        total_thrust_body = rotor_thrust.sum(dim=1)

        # Translate thrust to world frame and compute linear acceleration
        thrust_world = rotate_vector_by_quaternion(total_thrust_body, quaternion)
        linear_acc = thrust_world / mass + gravity

        # Integrate position and velocity
        new_velocity = velocity + linear_acc * dt
        new_position = position + new_velocity * dt

        # Angular dynamics
        I_omega = inertia * angular_velocity
        gyroscopic = torch.cross(angular_velocity, I_omega, dim=-1)
        angular_acc = inertia_inv * (torque_body - gyroscopic)
        new_angular_velocity = torch.clamp(angular_velocity + angular_acc * dt, -1e2, 1e2)

        # Quaternion integration
        omega_quat = torch.cat([
            torch.zeros_like(new_angular_velocity[..., :1]),
            new_angular_velocity,
        ], dim=-1)
        q_dot = 0.5 * quaternion_multiply(quaternion, omega_quat)
        new_quaternion = quaternion + q_dot * dt
        new_quaternion = new_quaternion / torch.norm(new_quaternion, dim=-1, keepdim=True)

        return new_rotor_speeds, new_position, new_velocity, new_quaternion, new_angular_velocity, total_thrust_body

    def _compute_observations(self) -> torch.Tensor:
        """Compute gate-relative observations for all envs."""
        gate_pos = self._gate_positions[self._target_gate_idx]  # (N, 3)
        gate_yaw = self._gate_yaws[self._target_gate_idx]        # (N,)

        cos_y = torch.cos(gate_yaw)
        sin_y = torch.sin(gate_yaw)

        # Position relative to target gate in gate frame
        rel_pos = self._position - gate_pos
        pos_gate_x = cos_y * rel_pos[:, 0] + sin_y * rel_pos[:, 1]
        pos_gate_y = -sin_y * rel_pos[:, 0] + cos_y * rel_pos[:, 1]
        pos_gate = torch.stack([pos_gate_x, pos_gate_y, rel_pos[:, 2]], dim=-1)

        # Velocity in gate frame
        vel_gate_x = cos_y * self._velocity[:, 0] + sin_y * self._velocity[:, 1]
        vel_gate_y = -sin_y * self._velocity[:, 0] + cos_y * self._velocity[:, 1]
        vel_gate = torch.stack([vel_gate_x, vel_gate_y, self._velocity[:, 2]], dim=-1)

        # Euler angles from quaternion (phi, theta) + yaw relative to gate
        w, x, y, z = (self._quaternion[:, i] for i in range(4))
        phi   = torch.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        theta = torch.asin(torch.clamp(2*(w*y - z*x), -1.0, 1.0))
        psi   = torch.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        yaw_rel = (psi - gate_yaw + torch.pi) % (2 * torch.pi) - torch.pi
        euler_gate = torch.stack([phi, theta, yaw_rel], dim=-1)

        # Rotor speeds normalized to [-1, 1]  (reference: (w/w_max)*2 - 1)
        rpm_scaled = self._rotor_speeds / self._max_rpm * 2.0 - 1.0

        obs_parts = [pos_gate, vel_gate, euler_gate, self._angular_velocity, rpm_scaled]

        # Future gate info
        for i in range(self.gates_ahead):
            next_idx = (self._target_gate_idx + i + 1) % self.num_gates
            gate_rel_pos = self._gate_pos_rel[next_idx]  # (N, 3)
            gate_rel_yaw = self._gate_yaw_rel[next_idx].unsqueeze(-1)  # (N, 1)
            obs_parts.append(gate_rel_pos)
            obs_parts.append(gate_rel_yaw)

        return torch.cat(obs_parts, dim=-1)

    def _detect_gate_events(
        self, pos_old: torch.Tensor, pos_new: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Detect gate passing, collision, and dead zone events.
        Returns: gate_passed, gate_collision, in_gate_deadzone — all (num_envs,) bool tensors.
        """
        gate_passed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        gate_collision = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        in_gate_deadzone = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        half_hole = self.gate_hole_size / 2.0
        half_outer = self.gate_outer_size / 2.0

        # Check target gate (offset 0) and previous gate (offset -1)
        for gate_offset in [0, -1]:
            is_target = gate_offset == 0
            gate_idx = (self._target_gate_idx + gate_offset) % self.num_gates

            gate_pos = self._gate_positions[gate_idx]  # (N, 3)
            gate_yaw = self._gate_yaws[gate_idx]        # (N,)

            cos_y = torch.cos(gate_yaw)
            sin_y = torch.sin(gate_yaw)

            rel_old = pos_old - gate_pos
            rel_new = pos_new - gate_pos

            # Forward axis (through gate plane)
            old_x = cos_y * rel_old[:, 0] + sin_y * rel_old[:, 1]
            new_x = cos_y * rel_new[:, 0] + sin_y * rel_new[:, 1]
            # Lateral axis
            new_y = -sin_y * rel_new[:, 0] + cos_y * rel_new[:, 1]
            # Vertical (z-up: no sign flip)
            new_z = rel_new[:, 2]

            crossed_fwd = (old_x < 0) & (new_x >= 0)
            crossed_rev = (old_x > 0) & (new_x <= 0)
            in_hole  = (torch.abs(new_y) < half_hole)  & (torch.abs(new_z) < half_hole)
            in_frame = (torch.abs(new_y) < half_outer) & (torch.abs(new_z) < half_outer)

            if is_target:
                gate_passed = crossed_fwd & in_hole
                in_gate_deadzone = (new_x > 0) & in_frame

            gate_collision |= (crossed_fwd & ~in_hole) | (crossed_rev & in_frame)

        # Gate passing overrides any collision (handles stacked/adjacent gates)
        gate_collision[gate_passed] = False

        return gate_passed, gate_collision, in_gate_deadzone

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
        if not isinstance(actions, torch.Tensor):
            actions = torch.tensor(actions, dtype=torch.float32, device=self.device)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        self._actions = actions.clamp(-1.0, 1.0)
        actions_0_1 = self._max_rpm * (self._actions + 1.0) / 2.0

        # Clone state tensors to break CUDAGraph output-buffer aliasing
        pos_old = self._position.clone()
        rotor_speeds = self._rotor_speeds.clone()
        position    = self._position.clone()
        velocity    = self._velocity.clone()
        quaternion  = self._quaternion.clone()
        angular_velocity = self._angular_velocity.clone()
        torch.compiler.cudagraph_mark_step_begin()

        # Physics step (compiled)
        (
            self._rotor_speeds,
            self._position,
            self._velocity,
            self._quaternion,
            self._angular_velocity,
            _total_thrust_body,
        ) = self._compiled_physics_step(
            actions_0_1,
            rotor_speeds, position, velocity, quaternion, angular_velocity,
            self._rotor_positions, self._thrust_directions, self._thrust_coefficients,
            self._rotor_torque_constants, self._rotor_torque_directions,
            self._rising_delay_constants, self._falling_delay_constants,
            self._mass, self._inertia, self._inertia_inv, self._gravity, self.dt,
        )

        # Gate events
        gate_passed, gate_collision, in_gate_deadzone = self._detect_gate_events(pos_old, self._position)

        # Termination conditions
        ground_collision = self._position[:, 2] < 0.0
        out_of_bounds = (
            (torch.abs(self._position[:, 0]) > 50.0) |
            (torch.abs(self._position[:, 1]) > 50.0) |
            (self._position[:, 2] > 7.5) |
            (torch.any(torch.abs(self._angular_velocity) > 1000.0, dim=1))
        )

        self.terminals = gate_collision | ground_collision | out_of_bounds
        self.truncations = self.episode_length_buf >= self.max_episode_length - 1

        # Rewards
        current_gate_pos = self._gate_positions[self._target_gate_idx]
        new_dist = torch.linalg.norm(self._position - current_gate_pos, dim=1)
        progress_reward = 10.0 * (self._prev_dist_to_gate - new_dist)
        rate_penalty = -0.001 * torch.linalg.norm(self._angular_velocity, dim=1)

        self.rewards = progress_reward + rate_penalty - 0.02
        self.rewards[in_gate_deadzone & ~gate_passed] = -0.1
        self.rewards[gate_passed] += 10.0
        self.rewards[gate_collision | ground_collision | out_of_bounds] = -10.0

        # Advance target gate for envs that passed
        self._target_gate_idx[gate_passed] = (self._target_gate_idx[gate_passed] + 1) % self.num_gates

        # Update prev distance (to next gate for envs that just passed)
        next_gate_pos = self._gate_positions[self._target_gate_idx]
        self._prev_dist_to_gate = torch.linalg.norm(self._position - next_gate_pos, dim=1)

        # Episode tracking
        self.episode_length_buf += 1
        self._cumulative_rewards += self.rewards

        # Auto-reset
        if self.autoreset:
            reset_envs = torch.where(self.terminals | self.truncations)[0]
            if len(reset_envs) > 0:
                self._completed_episode_lengths[reset_envs] = self.episode_length_buf[reset_envs].float()
                self._completed_episode_rewards[reset_envs] = self._cumulative_rewards[reset_envs]
                self._reset_idx(reset_envs)

        # Render
        if self.render_mode == "human":
            self._render()

        self.observations = self._compute_observations()

        info = {
            "mean_reward": self.rewards.mean().item(),
            "gates_passed": gate_passed.float().mean().item(),
            "gate_collisions": gate_collision.float().mean().item(),
            "ground_collisions": ground_collision.float().mean().item(),
            "episode_length_mean": self._completed_episode_lengths.mean().item(),
            "episode_reward_mean": self._completed_episode_rewards.mean().item(),
        }
        self.infos = [info]
        return self.observations, self.rewards, self.terminals, self.truncations, self.infos

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(env_ids)
        self.observations = self._compute_observations()
        return self.observations, [{}]

    def _reset_idx(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        num_reset = len(env_ids)
        self.episode_length_buf[env_ids] = 0
        self._rotor_speeds[env_ids] = 0.0
        self._cumulative_rewards[env_ids] = 0.0

        if self.initialize_at_random_gates:
            gate_idx = torch.randint(0, self.num_gates, (num_reset,), device=self.device)
            self._target_gate_idx[env_ids] = gate_idx
            gate_pos = self._gate_positions[gate_idx]   # (num_reset, 3)
            gate_yaw = self._gate_yaws[gate_idx]         # (num_reset,)
            # Place 1m behind gate plane (negative forward direction)
            offset = torch.stack([torch.cos(gate_yaw), torch.sin(gate_yaw), torch.zeros_like(gate_yaw)], dim=-1)
            self._position[env_ids] = gate_pos - offset
        else:
            self._target_gate_idx[env_ids] = 0
            self._position[env_ids] = self._start_position

        # Small random velocity perturbation
        self._velocity[env_ids] = torch.zeros(num_reset, 3, device=self.device).uniform_(-0.5, 0.5)
        self._quaternion[env_ids] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        self._angular_velocity[env_ids] = torch.zeros(num_reset, 3, device=self.device).uniform_(-0.1, 0.1)

        # Dynamics randomization
        delta = self.dynamics_randomization_delta
        if delta > 0:
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

        # Initialize prev_dist_to_gate
        gate_pos = self._gate_positions[self._target_gate_idx[env_ids]]
        self._prev_dist_to_gate[env_ids] = torch.linalg.norm(self._position[env_ids] - gate_pos, dim=1)

    def _render(self):
        if not self._gates_logged:
            gate_info = [
                (self._gate_positions[i].cpu().numpy(), float(self._gate_yaws[i].cpu()))
                for i in range(self.num_gates)
            ]
            log_gates(gate_info)
            self._gates_logged = True

        position = self._position[0].detach().cpu().numpy()
        quaternion = self._quaternion[0].detach().cpu().numpy()  # (w, x, y, z)
        quat_xyzw = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
        log_drone_pose(position, quat_xyzw)

        rr.log("race/target_gate", rr.Scalars(float(self._target_gate_idx[0])))
        rr.log("race/reward", rr.Scalars(float(self.rewards[0])))
        rr.log("race/dist_to_gate", rr.Scalars(float(self._prev_dist_to_gate[0])))

    def close(self):
        pass
