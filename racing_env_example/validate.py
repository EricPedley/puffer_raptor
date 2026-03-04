from cyclone.simulation.rl.quad_race_env import Quadcopter3DGates
from cyclone.simulation.rl.randomization import randomization_fixed_params_5inch
from cyclone.shared.logging_utils import (
    log_gates,
    log_drone_pose,
    log_velocity,
    fix_viz_rot,
)
from cyclone.shared.gate_maps import (
    gate_positions as hard_pos,
    gate_yaws as hard_yaw,
    racetrack_start as hard_start,
)
import rerun as rr
from stable_baselines3 import PPO
from scipy.spatial.transform import Rotation
import numpy as np
from pathlib import Path
import subprocess
import os

file_path = Path(__file__).parent


def test_rotation_conventions(drone_euler):
    """Test different rotation conventions to see which matches the drone's actual orientation"""

    phi, theta, psi = drone_euler

    print(f"Drone Euler angles: phi={phi:.3f}, theta={theta:.3f}, psi={psi:.3f}")

    # Manually construct the rotation matrix as in the quadcopter dynamics
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]
    )

    Ry = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )

    Rz = np.array(
        [[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]]
    )

    # The rotation matrix from the dynamics: R = Rz*Ry*Rx
    R_manual = Rz @ Ry @ Rx

    print("Manual rotation matrix (Rz*Ry*Rx):")
    print(R_manual)

    # Test different scipy conventions
    conventions = [
        "XYZ",
        "XZY",
        "YXZ",
        "YZX",
        "ZXY",
        "ZYX",
        "xyz",
        "xzy",
        "yxz",
        "yzx",
        "zxy",
        "zyx",
    ]

    for convention in conventions:
        try:
            if convention.isupper():  # Extrinsic
                R_scipy = Rotation.from_euler(
                    convention, [phi, theta, psi], degrees=False
                ).as_matrix()
            else:  # Intrinsic
                R_scipy = Rotation.from_euler(
                    convention, [phi, theta, psi], degrees=False
                ).as_matrix()

            # Check if matrices are close
            if np.allclose(R_manual, R_scipy, atol=1e-6):
                print(f"MATCH FOUND: {convention}")
                return convention

            # Also check transpose (in case it's the inverse transformation)
            if np.allclose(R_manual, R_scipy.T, atol=1e-6):
                print(f"TRANSPOSE MATCH FOUND: {convention} (use inverse)")
                return convention + "_inverse"

        except Exception as e:
            print(f"Error with {convention}: {e}")

    print("No exact match found")
    return None


def main():
    model_path = file_path / "models" / "general_session" / "general_model"

    try:
        most_recent_model = max(model_path.glob("*"), key=os.path.getmtime)
        print(f"Most recent file: {most_recent_model}")
        model_path = most_recent_model
    except ValueError:
        print("No files found in the models folder")
        model_path = file_path / "OptimizedModel"

    model = PPO.load(str(model_path))
    env = Quadcopter3DGates(
        num_envs=1,
        randomization=randomization_fixed_params_5inch,
        gates_pos=hard_pos,
        gate_yaw=hard_yaw,
        start_pos=hard_start,
        initialize_at_random_gates=False,
    )
    obs = env.reset()

    trajectory_points = []
    step_count = 0

    exitCondition = False

    log_gates(list(zip(env.gate_pos, env.gate_yaw)))

    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if done[0]:
            # Debug - Jacob
            exitCondition = True
            print("Terminated: ", done[0])
            print("")
            report = env.return_dones_report()
            print("max steps: ", report[0])
            print("ground col: ", report[1])
            print("out of bounds: ", report[2])
            print("gate col: ", report[3])
            print("")
            #
            break

        # Extract drone state from world_states (first environment) - this is the correct source
        drone_position = env.world_states[0, 0:3]  # [x, y, z] in world frame
        drone_velocity = env.world_states[0, 3:6]  # [vx, vy, vz] in world frame
        drone_euler = env.world_states[0, 6:9]  # [phi, theta, psi] in world frame

        phi, theta, psi = drone_euler

        rotation_quat = Rotation.from_euler("xyz", drone_euler, degrees=False).as_quat()

        log_drone_pose(position=drone_position, quaternion=rotation_quat)
        rr.log(
            "drone/drone_axis",
            rr.Transform3D(
                translation=fix_viz_rot.apply(drone_position),
                quaternion=(fix_viz_rot * Rotation.from_quat(rotation_quat)).as_quat(),
                axis_length=1,
            ),
        )

        rr.log("drone/phi", rr.Scalars(drone_euler[0]))
        rr.log("drone/theta", rr.Scalars(drone_euler[1]))
        rr.log("drone/psi", rr.Scalars(drone_euler[2]))

        # Add to trajectory and log it
        trajectory_points.append(fix_viz_rot.apply(drone_position.copy()))
        rr.log(
            "drone/trajectory", rr.LineStrips3D([trajectory_points], colors=[0, 255, 0])
        )

        # Log velocity vector
        if np.linalg.norm(drone_velocity) > 0.1:  # Only show if moving
            log_velocity(
                position=drone_position,
                velocity=drone_velocity,
                model_name="drone/velocity",
            )
        # Log gates
        for i, (gate_pos, gate_yaw) in enumerate(zip(env.gate_pos, env.gate_yaw)):
            color = (
                [0, 255, 0] if i == env.target_gates[0] else [128, 128, 128]
            )  # Green for target, gray for others

            # Log gate center
            rr.log(
                f"gates/gate_{i}/position",
                rr.Points3D(
                    [fix_viz_rot.apply(gate_pos)], colors=[color], radii=[0.05]
                ),
            )  # .75 radii for gate size

            # Log gate orientation (as a small arrow)
            gate_direction = np.array([np.cos(gate_yaw), np.sin(gate_yaw), 0]) * 0.5
            rr.log(
                f"gates/gate_{i}/direction",
                rr.Arrows3D(
                    origins=[fix_viz_rot.apply(gate_pos)],
                    vectors=[fix_viz_rot.apply(gate_direction)],
                    colors=[color],
                ),
            )

        # Log metrics
        rr.log("metrics/reward", rr.Scalars(reward[0]))
        rr.log("metrics/step", rr.Scalars(step_count))
        rr.log(
            "metrics/distance_to_target",
            rr.Scalars(
                np.linalg.norm(drone_position - env.gate_pos[env.target_gates[0]])
            ),
        )

        # Log motor speeds
        motor_speeds = env.world_states[0, 12:16]
        for i, speed in enumerate(motor_speeds):
            rr.log(f"motors/motor_{i+1}", rr.Scalars(speed))

        # Log collision events
        if info[0].get("gate_passed", False):
            rr.log(
                "events/gate_passed",
                rr.TextLog(f"Gate {env.target_gates[0]-1} passed!"),
            )

        if info[0].get("gate_collision", False):
            rr.log("events/collision", rr.TextLog("Gate collision!"))

        if info[0].get("ground_collision", False):
            rr.log("events/collision", rr.TextLog("Ground collision!"))

        step_count += 1

    # Debug - Jacob
    if not exitCondition:
        print("")
        print("END OF TIME")
        print("")

    env.close()


if __name__ == "__main__":
    # init rerun
    rr.init("quadcopter_controller_validation", spawn=False)
    rr.save("quadcopter_validation.rrd")
    main()
    subprocess.Popen(["rerun", "quadcopter_validation.rrd"])
