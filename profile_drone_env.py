"""
Profile _physics_step_impl with and without torch.compile.

Usage:
    uv run profile_drone_env.py [--num-envs N] [--steps K] [--compile] [--mode MODE]
"""
import argparse
import time
import torch

from drone_env import QuadcopterEnv


def make_random_inputs(num_envs: int, device: torch.device):
    N = num_envs
    actions_0_1      = torch.rand(N, 4, device=device)
    actions_clamped  = torch.rand(N, 4, device=device)
    last_action      = torch.rand(N, 4, device=device)
    rotor_speeds     = torch.rand(N, 4, device=device)
    position         = torch.randn(N, 3, device=device) * 0.1
    velocity         = torch.randn(N, 3, device=device) * 0.5
    # Identity quaternion + small perturbation, then normalise
    quaternion       = torch.zeros(N, 4, device=device)
    quaternion[:, 0] = 1.0
    quaternion       = quaternion + torch.randn_like(quaternion) * 0.01
    quaternion       = quaternion / quaternion.norm(dim=-1, keepdim=True)
    angular_velocity = torch.randn(N, 3, device=device) * 0.5
    return (
        actions_0_1, actions_clamped, last_action, rotor_speeds,
        position, velocity, quaternion, angular_velocity,
    )


def run_benchmark(fn, env, inputs, steps: int, label: str):
    (actions_0_1, actions_clamped, last_action, rotor_speeds,
     position, velocity, quaternion, angular_velocity) = inputs

    # Warmup
    for _ in range(5):
        fn(
            env,
            actions_0_1, actions_clamped, last_action, rotor_speeds,
            position, velocity, quaternion, angular_velocity,
            env._rotor_positions,
            env._thrust_directions,
            env._thrust_coefficients,
            env._rotor_torque_constants,
            env._rotor_torque_directions,
            env._rising_delay_constants,
            env._falling_delay_constants,
            env._mass,
            env._inertia,
            env._inertia_inv,
            env._gravity,
            env.dt,
            env._decimation_steps,
            env._max_rpm,
            env.rwd_scale,
            env.rwd_constant,
            env.rwd_termination_penalty,
            env.rwd_position,
            env.rwd_orientation,
            env.rwd_linear_velocity,
            env.rwd_angular_velocity,
            env.rwd_action,
            env.rwd_d_action,
            env.rwd_non_negative,
            env.term_position,
            env.term_linear_velocity,
            env.term_angular_velocity,
        )
    if env.device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        fn(
            env,
            actions_0_1, actions_clamped, last_action, rotor_speeds,
            position, velocity, quaternion, angular_velocity,
            env._rotor_positions,
            env._thrust_directions,
            env._thrust_coefficients,
            env._rotor_torque_constants,
            env._rotor_torque_directions,
            env._rising_delay_constants,
            env._falling_delay_constants,
            env._mass,
            env._inertia,
            env._inertia_inv,
            env._gravity,
            env.dt,
            env._decimation_steps,
            env._max_rpm,
            env.rwd_scale,
            env.rwd_constant,
            env.rwd_termination_penalty,
            env.rwd_position,
            env.rwd_orientation,
            env.rwd_linear_velocity,
            env.rwd_angular_velocity,
            env.rwd_action,
            env.rwd_d_action,
            env.rwd_non_negative,
            env.term_position,
            env.term_linear_velocity,
            env.term_angular_velocity,
        )
    if env.device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    print(f"[{label}] {steps} steps in {elapsed:.3f}s  "
          f"({1e3 * elapsed / steps:.3f} ms/step, "
          f"{steps / elapsed:.1f} steps/s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--steps",    type=int, default=500)
    parser.add_argument("--compile",  action="store_true",
                        help="benchmark compiled version (in addition to eager)")
    parser.add_argument("--mode",     default="default",
                        help="torch.compile mode (default, max-autotune, …)")
    parser.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"device={args.device}  num_envs={args.num_envs}  steps={args.steps}")

    # Build an env so we can borrow all the physics constants from it.
    # use_compile=False so we control compilation ourselves below.
    env = QuadcopterEnv(
        num_envs=args.num_envs,
        device=args.device,
        use_compile=False,
    )
    env.reset()

    inputs = make_random_inputs(args.num_envs, env.device)

    # --- Eager ---
    run_benchmark(
        QuadcopterEnv._physics_step_impl,
        env, inputs, args.steps,
        label="eager",
    )

    # --- Compiled ---
    if args.compile:
        compiled_fn = torch.compile(
            QuadcopterEnv._physics_step_impl,
            mode=args.mode,
            fullgraph=False,
        )
        run_benchmark(
            compiled_fn,
            env, inputs, args.steps,
            label=f"compiled({args.mode})",
        )


if __name__ == "__main__":
    main()
