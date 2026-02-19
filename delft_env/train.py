# ruff: noqa
# library imports
import os
import sys
from stable_baselines3 import PPO
from datetime import datetime
from stable_baselines3.common.vec_env import VecMonitor
import numpy as np

# custom imports
from quad_race_env import *
from randomization import *
from cyclone.shared.gate_maps import (
    gate_positions as hard_pos,
    gate_yaws as hard_yaw,
    racetrack_start as hard_start,
    positions_with_extr_gate as easy_pos,
    yaws_with_extra_gate as easy_yaw,
    easy_start,
)
# from validate import run_validation

import argparse

parser = argparse.ArgumentParser(description="Training session configuration")

# Name of the training session
parser.add_argument("session_name", type=str, help="Name of the training session")

# Name of the model
parser.add_argument("name", type=str, help="Name of the model")

# Architecture of the policy (list of integers)
parser.add_argument(
    "--pi",
    type=int,
    nargs="+",
    default=[64, 64],
    help="Architecture of the policy (e.g., --pi 64 64). Default is [64, 64]",
)

# Architecture of the value function (list of integers)
parser.add_argument(
    "--vf",
    type=int,
    nargs="+",
    default=[64, 64],
    help="Architecture of the value function (e.g., --vf 64 64). Default is [64, 64]",
)

# State history input length (default is 0)
parser.add_argument(
    "--state_history",
    type=int,
    default=0,
    help="State history input length (default is 0)",
)

# Action history input length (default is 0)
parser.add_argument(
    "--action_history",
    type=int,
    default=0,
    help="Action history input length (default is 0)",
)

# History step size (default is 1)
parser.add_argument(
    "--history_step_size", type=int, default=1, help="History step size (default is 1)"
)

# Param input (boolean, default is False)
parser.add_argument("--param_input", action="store_true", help="Use parameter input")

# Param input noise (default = 0.0)
parser.add_argument(
    "--param_input_noise", type=float, default=0.0, help="Parameter input noise"
)

# Randomization (randomized, fixed_5inch, fixed_3inch)
parser.add_argument(
    "--randomization",
    type=str,
    default="randomized",
    help="Randomization (randomized, fixed_5inch, fixed_3inch)",
)

# Load Model
parser.add_argument(
    "--load_model",
    type=str,
    default=None,
    help="Path to existing model to continue training from (e.g., models/session1/model_name/100000000.zip)",
)

# Parse the arguments
args = parser.parse_args()

# print summary of the arguments
print("Training session configuration:")
print(f"Session name: {args.session_name}")
print(f"Model name: {args.name}")
print(f"Policy architecture: {args.pi}")
print(f"Value function architecture: {args.vf}")
print(f"State history input length: {args.state_history}")
print(f"Action history input length: {args.action_history}")
print(f"History step size: {args.history_step_size}")
print(f"Parameter input: {args.param_input}")
print(f"Parameter input noise: {args.param_input_noise}")
print(f"Randomization: {args.randomization}")

# DEFINE RACE TRACK
#         TU Delft Path
"""
r = 1.5
gate_pos = np.array([
    [ r,  -r, -1.5],
    [ 0,   0, -1.5],
    [-r,   r, -1.5],
    [ 0, 2*r, -1.5],
    [ r,   r, -1.5],
    [ 0,   0, -1.5],
    [-r,  -r, -1.5],
    [ 0,-2*r, -1.5]
])
gate_yaw = np.array([1,2,1,0,-1,-2,-1,0])*np.pi/2
start_pos = gate_pos[0] + np.array([0,-1.,0])
"""


# SETUP LOGGING
models_dir = "models/" + args.session_name
log_dir = "logs/" + args.session_name
video_log_dir = "videos/" + args.session_name

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
if not os.path.exists(video_log_dir):
    os.makedirs(video_log_dir)

# Date and time string for unique folder names
datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")

# CREATE ENVIRONMENTS
if args.randomization == "randomized":
    randomization = randomization_big
elif args.randomization == "fixed_5inch":
    randomization = randomization_fixed_params_5inch
elif args.randomization == "5inch_10_percent":
    randomization = randomization_5inch_10_percent
elif args.randomization == "5inch_20_percent":
    randomization = randomization_5inch_20_percent
elif args.randomization == "5inch_30_percent":
    randomization = randomization_5inch_30_percent
elif args.randomization == "fixed_3inch":
    randomization = randomization_fixed_params_3inch
elif args.randomization == "3inch_10_percent":
    randomization = randomization_3inch_10_percent
elif args.randomization == "3inch_20_percent":
    randomization = randomization_3inch_20_percent
elif args.randomization == "3inch_30_percent":
    randomization = randomization_3inch_30_percent
else:
    print("Randomization not recognized")
    # kill the process
    sys.exit()


def create_env(gate_pos, gate_yaw, start_pos, num_envs=100):
    env = Quadcopter3DGates(
        num_envs=num_envs,
        gates_pos=gate_pos,
        gate_yaw=gate_yaw,
        start_pos=start_pos,
        randomization=randomization,
        gates_ahead=1,
        num_state_history=args.state_history,
        num_action_history=args.action_history,
        history_step_size=args.history_step_size,
        param_input=args.param_input,
        param_input_noise=args.param_input_noise,
        initialize_at_random_gates=True,
    )
    return VecMonitor(env)


test_env = create_env(
    gate_pos=easy_pos, gate_yaw=easy_yaw, start_pos=easy_start, num_envs=1
)

easy_env = create_env(gate_pos=easy_pos, gate_yaw=easy_yaw, start_pos=easy_start)
hard_env = create_env(gate_pos=hard_pos, gate_yaw=hard_yaw, start_pos=hard_start)
# MODEL DEFINITION

policy_kwargs = dict(
    activation_fn=torch.nn.ReLU, net_arch=[dict(pi=args.pi, vf=args.vf)], log_std_init=0
)

if args.load_model:
    print(f"Loading existing model from: {args.load_model}")
    model = PPO.load(args.load_model, env=hard_env)  # Uses hard env
    model.ent_coef = 0.0
    print("Model loaded successfully!")
else:
    model = PPO(
        "MlpPolicy",
        hard_env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=log_dir,
        n_steps=1000,
        batch_size=5000,
        n_epochs=10,
        gamma=0.999,
        ent_coef=0.0,
    )

print(
    "Model created with policy architecture",
    args.pi,
    "and value function architecture",
    args.vf,
)
print("-----------------------------------")
print(model.policy)
print("-----------------------------------")
print("Logging to", log_dir)
print("Saving models to", models_dir)
print("Saving videos to", video_log_dir)

# TESTING
test_env.reset()

# do 100 steps and print state and action
for i in range(100):
    if i % 20 == 0:
        print("step", i)
        # num = test_env.num_state_history+1
        # state_len = int(len(test_env.states[0])/num)
        # for j in range(num):
        #    print('state', j, '=', test_env.states[0][j*state_len:(j+1)*state_len])
        actions, _ = model.predict(test_env.states, deterministic=True)
        states, rewards, dones, infos = test_env.step(actions)
        print("actions=", actions[0])


# TRAINING
# training loop saves model every 10 policy rollouts and saves a video animation
def train(model, log_name, n=int(2e8)):
    TIMESTEPS = model.n_steps * model.env.num_envs * 10
    n += model.num_timesteps  # updates traget for loading models

    while model.num_timesteps < n:
        model.learn(
            total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=log_name
        )
        time_steps = model.num_timesteps
        # save model
        model.save(models_dir + "/" + log_name + "/" + str(time_steps))
        print("Model saved at", models_dir + "/" + log_name + "/" + str(time_steps))

        # show progress as model trains
        # subprocess.Popen(['python', 'validate.py'])


# name = 'figure8_64_64_again!'
# import shutil
# shutil.rmtree(log_dir + '/' + name + '_0', ignore_errors=True)
# shutil.rmtree(models_dir + '/' + name, ignore_errors=True)
# shutil.rmtree(video_log_dir + '/' + name, ignore_errors=True)

# RUN TRAINING LOOP
name = args.name

# check if model already exists
# if os.path.exists(models_dir + '/' + name):
#     print(f"Model {name} already exists. Do you want to overwrite it (this will delete the existing model/logs/videos)? (y/n)")

import shutil

if os.path.exists(log_dir + "/" + name + "_0"):
    print("Deleting logs...")
    shutil.rmtree(log_dir + "/" + name + "_0", ignore_errors=True)
if os.path.exists(models_dir + "/" + name):
    print("Deleting models...")
    shutil.rmtree(models_dir + "/" + name, ignore_errors=True)
if os.path.exists(video_log_dir + "/" + name):
    print("Deleting videos...")
    shutil.rmtree(video_log_dir + "/" + name, ignore_errors=True)

print("Training model", name)
train(model, name)
