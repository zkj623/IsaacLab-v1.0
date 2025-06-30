# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
# parser.add_argument("--cpu", action="store_true", default=False, help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
# print(app_launcher.app)
simulation_app = app_launcher.app

import omni

# ext_manager = omni.kit.app.get_app().get_extension_manager()
# ext_manager.set_extension_enabled_immediate("omni.isaac.ros2_bridge", True)


"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import numpy as np
import heapq
import copy
import carb

from rsl_rl.runners import OnPolicyRunner

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab.utils import convert_dict_to_backend
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    PushEnvWrapper,
    WalkEnvWrapper,
    ClimbEnvWrapper,
    NavEnvWrapper,
    OriginalHighEnvWrapper,
    OriginalLowEnvWrapper,
    OriginalEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.game_env_cfg as custom_rl_env
from omni.isaac.lab.managers import CommandTerm
from omni.isaac.lab.utils.math import combine_frame_transforms, subtract_frame_transforms, euler_xyz_from_quat
from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file, get_viewport_from_window_name, get_active_viewport_and_window

import omni.graph.core as og
import omni.replicator.core as rep
from omni.isaac.lab.sensors.camera import Camera, CameraCfg
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
from omni.isaac.lab.markers.config import RAY_CASTER_MARKER_CFG
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

# from omni.isaac.core_nodes import IsaacCreateViewport

from openai import OpenAI
import openai
import requests
import time
import json

from collections import defaultdict
import math
import ast
from graphviz import Digraph
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from ultralytics import YOLO
import usdrt.Sdf

from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration

from transformers import CLIPProcessor, CLIPModel

from LLM_planning import LLM_task_planning
from scipy.ndimage import binary_dilation

model = YOLO('yolov8m.pt')

torch.set_printoptions(sci_mode=False, precision=4)

API_SECRET_KEY = "sk-zk2a34977de3d68b2fb0058cfe0b531907e66b5db20733c7"
BASE_URL = "https://api.zhizengzeng.com/v1/"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# processor_clip = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# #.to("cuda")

# processor_blip = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
# model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to("cuda")

# results = model('https://ultralytics.com/images/bus.jpg')
# results = model('/home/zkj/IsaacLab/rgb_depth/result.png')

# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     print(boxes)
#     result.save(filename="result.jpg")  # save to disk

# chat
def chat_completions(query):
    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
    resp = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
    )
    return resp.choices[0].message.content, resp.usage.prompt_tokens, resp.usage.completion_tokens


def sub_keyboard_event(event, *args, **kwargs) -> bool:

    if len(custom_rl_env.base_command) > 0:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == 'W':
                custom_rl_env.base_command["0"] = [1, 0, 0]
            if event.input.name == 'S':
                custom_rl_env.base_command["0"] = [-1, 0, 0]
            if event.input.name == 'A':
                custom_rl_env.base_command["0"] = [0, 1, 0]
            if event.input.name == 'D':
                custom_rl_env.base_command["0"] = [0, -1, 0]
            if event.input.name == 'Q':
                custom_rl_env.base_command["0"] = [0, 0, 1]
            if event.input.name == 'E':
                custom_rl_env.base_command["0"] = [0, 0, -1]

            if len(custom_rl_env.base_command) > 1:
                if event.input.name == 'I':
                    custom_rl_env.base_command["1"] = [1, 0, 0]
                if event.input.name == 'K':
                    custom_rl_env.base_command["1"] = [-1, 0, 0]
                if event.input.name == 'J':
                    custom_rl_env.base_command["1"] = [0, 1, 0]
                if event.input.name == 'L':
                    custom_rl_env.base_command["1"] = [0, -1, 0]
                if event.input.name == 'U':
                    custom_rl_env.base_command["1"] = [0, 0, 1]
                if event.input.name == 'O':
                    custom_rl_env.base_command["1"] = [0, 0, -1]
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            for i in range(len(custom_rl_env.base_command)):
                custom_rl_env.base_command[str(i)] = [0, 0, 0]
    return True


def blip(img_path, processor, model):
    raw_image = Image.open(img_path).convert('RGB')

    # conditional image captioning
    # text = "You are a quadruped robot. Help me describe what you see in the picture."
    text = "the road"
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    # >>> a photography of a woman and her dog

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))

def clip(img_path, processor, model):
    TOKENIZERS_PARALLELISM = False

    # processor = CLIPProcessor.from_pretrained("./pretrained_models/clip-vit-base-patch32")
    # model = CLIPModel.from_pretrained("./pretrained_models/clip-vit-base-patch32", torch_dtype=torch.float16).to("cuda")

    raw_image = Image.open(img_path).convert('RGB')

    text=["The road ahead has no boxes.", "The road ahead is completely blocked by boxes.", "The high place requires a box to be reached."]
    inputs = processor(text, images=raw_image, return_tensors="pt", padding=True)
    #.to("cuda")

    outputs = model(**inputs)

    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    # print(logits_per_image)
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    print(probs)
    max_index = probs[0].argmax()
    print("------------image reasoning-------------")
    print("------------------",text[max_index])
    
    # if probs[0][0] > probs[0][1]:
    #     print(text[0])
    # else:
    #     print(text[1])

    return max_index



def define_markers() -> VisualizationMarkers:
    """Define markers with various different shapes."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/myMarkers",
        markers={
            "arrow_x": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(0.2, 0.2, 0.4),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "cube": sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.8, 0.0)),
            ),
            "sphere": sim_utils.SphereCfg(
                radius=0.15,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "small_sphere": sim_utils.SphereCfg(
                radius=0.10,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
            "cylinder": sim_utils.CylinderCfg(
                radius=0.5,
                height=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
            "cone": sim_utils.ConeCfg(
                radius=0.5,
                height=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)

# create markers
my_visualizer = define_markers()


def init_sim():

    # acquire input interface
    _input = carb.input.acquire_input_interface()
    _appwindow = omni.appwindow.get_default_app_window()
    _keyboard = _appwindow.get_keyboard()
    _sub_keyboard = _input.subscribe_to_keyboard_events(_keyboard, sub_keyboard_event)

    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # print(env_cfg.scene.num_envs)

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # print(env_cfg.observations)
    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # Set main camera
    # env.sim.set_camera_view([-10.0, -10.0, 8.0], [-2.0, -2.0, 0.0])
    # env.sim.set_camera_view([0.0, -4.0, 12.0], [-0.0, -0.2, 0.0])
    env.sim.set_camera_view([0.0, -0.2, 15.0], [-0.0, -0.0, 0.0])
    # env.sim.set_camera_view([0.0, -8.0, 8.0], [-0.0, -0.0, 0.0])
    # print(env.ui_window_class_type)
    # env.unwrapped.cfg.observations_backup = copy.deepcopy(env.unwrapped.cfg.observations)
    # env0 = RslRlVecEnvWrapper(env)

    # print("env: ", env.unwrapped.cfg.observations.policy)
    # print("env: ", env.unwrapped.cfg.actions.pre_trained_policy_action.low_level_observations.height_scan)
    # print("action: ", env.unwrapped.action_manager)
    # env_copy = copy.deepcopy(env)

    env1 = PushEnvWrapper(env)
    # print("env: ", env.unwrapped.cfg.observations.policy)
    # print("env1: ", env1.unwrapped.cfg.observations.policy)
    # print(env1.observation_space)
    # print(env1.unwrapped.observation_manager.group_obs_dim["policy"][0])
    
    # wrap around environment for rsl-rl
    env1 = RslRlVecEnvWrapper(env1)
    # print(env1.observation_space)
    # print(env1.num_obs)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # load previously trained model
    ppo_runner = OnPolicyRunner(env1, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(
        ppo_runner.alg.actor_critic, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt"
    )
    export_policy_as_onnx(
        ppo_runner.alg.actor_critic, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )

    # next policy inference
    env2 = WalkEnvWrapper(env)
    # print("env2: ", env2.unwrapped.cfg.observations.policy)
    # print("env2: ", env2.unwrapped.cfg.actions.pre_trained_policy_action.low_level_observations.height_scan)

    env2 = RslRlVecEnvWrapper(env2)

    args_cli2 = args_cli
    args_cli2.task = "Isaac-Navigation-Flat-Unitree-Go2-v0"

    agent_cfg2: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli2.task, args_cli2)

    # # specify directory for logging experiments
    log_root_path2 = os.path.join("logs", "rsl_rl", "unitree_go2_navigation")
    log_root_path2 = os.path.abspath(log_root_path2)
    # resume_path2 = get_checkpoint_path(log_root_path2, "stair_navigation", "model_250.pt")
    resume_path2 = get_checkpoint_path(log_root_path2, "2025-01-17_22-35-04", "model_600.pt")

    # load previously trained model
    ppo_runner2 = OnPolicyRunner(env2, agent_cfg2.to_dict(), log_dir=None, device=agent_cfg2.device)
    ppo_runner2.load(resume_path2)

    # obtain the trained policy for inference
    policy2 = ppo_runner2.get_inference_policy(device=env.unwrapped.device)

    env3 = NavEnvWrapper(env)
    # print("env2: ", env2.unwrapped.cfg.observations.policy)
    # print("env2: ", env2.unwrapped.cfg.actions.pre_trained_policy_action.low_level_observations.height_scan)

    env3 = RslRlVecEnvWrapper(env3)

    args_cli3 = args_cli
    args_cli3.task = "Isaac-Navigation-Flat-Unitree-Go2-v0"

    agent_cfg3: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli3.task, args_cli3)

    # # specify directory for logging experiments
    log_root_path3 = os.path.join("logs", "rsl_rl", "unitree_go2_navigation")
    log_root_path3 = os.path.abspath(log_root_path3)
    resume_path3 = get_checkpoint_path(log_root_path3, "nav_obstacle", "model_1499.pt")

    # load previously trained model
    ppo_runner3 = OnPolicyRunner(env3, agent_cfg3.to_dict(), log_dir=None, device=agent_cfg3.device)
    ppo_runner3.load(resume_path3)

    # obtain the trained policy for inference
    policy3 = ppo_runner3.get_inference_policy(device=env.unwrapped.device)

    env4 = ClimbEnvWrapper(env)
    # print("env2: ", env2.unwrapped.cfg.observations.policy)
    # print("env2: ", env2.unwrapped.cfg.actions.pre_trained_policy_action.low_level_observations.height_scan)

    env4 = RslRlVecEnvWrapper(env4)

    args_cli4 = args_cli
    args_cli4.task = "Isaac-Velocity-Rough-Unitree-Go2-v0"

    agent_cfg4: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli4.task, args_cli4)

    # # specify directory for logging experiments
    log_root_path4 = os.path.join("logs", "rsl_rl", "unitree_go2_rough")
    log_root_path4 = os.path.abspath(log_root_path4)
    # resume_path4 = get_checkpoint_path(log_root_path4, "climb_stable", "model_8100.pt")
    resume_path4 = get_checkpoint_path(log_root_path4, "2025-01-18_23-07-21", "model_9700.pt")

    # load previously trained model
    ppo_runner4 = OnPolicyRunner(env4, agent_cfg4.to_dict(), log_dir=None, device=agent_cfg4.device)
    ppo_runner4.load(resume_path4)

    # obtain the trained policy for inference
    policy4 = ppo_runner4.get_inference_policy(device=env.unwrapped.device)

    env0 = OriginalHighEnvWrapper(env)
    env0 = RslRlVecEnvWrapper(env0)

    # print("env: ", env)
    # print("env1: ", env1)
    # print("env2: ", env2)

    # Create a dictionary to store policies
    policies = {
        "walk": policy2,
        "push": policy,
        "navigation": policy3,
        "climb": policy4
    }

    return env0, env, policies

def get_object_description(env, objects_all):
    visible_objects = []

    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[:, :2]
    robot_pos = robot_pos[0]
    robot_heading = env.unwrapped.scene["robot"].data.heading_w
    # print("robot_pos: ", robot_pos)

    # Assuming env provides a method to get objects and their properties
    objects = env.unwrapped.scene.rigid_objects
    # print("objects: ", objects)

    # Define the vertices of obstacles
    half_side = 1.0 / 2
    center_x, center_y = -3.5, -1.5
    vertices = [
        (center_x - half_side, center_y - half_side),  
        (center_x + half_side, center_y - half_side),  
        (center_x + half_side, center_y + half_side),  
        (center_x - half_side, center_y + half_side)   
    ]

    cube_obstacle = [
        (vertices[0], vertices[1]),  # bottom
        (vertices[1], vertices[2]),  # right
        (vertices[2], vertices[3]),  # top
        (vertices[3], vertices[0])   # left
    ]

    obstacles = cube_obstacle


    for obj_name, obj in objects.items():
        # Assuming each object has properties: center, size, movable, on_top_of
        object_pos = obj.data.root_pos_w[:, :2]  # Replace with actual method to get center
        object_pos = object_pos[0]
        size = obj.cfg.spawn.size     # Replace with actual method to get size
        movable = "movable"
        on_top_of = "ground"       

        # Check if the object is within the robot's observation range
        if is_within_observation_range(robot_pos, robot_heading, object_pos, obstacles):  # Implement this check
            object_pos = object_pos.tolist()
            object_pos = [round(num, 2) for num in object_pos]
            description = f"* [{obj_name}]: <center>: {object_pos}; <size>: {size}. <movable>: {movable}; <on_top_of>: {on_top_of}"
            visible_objects.append(description)

            # if obj_name not in objects_all.keys():
            objects_all[obj_name] = description

    return "\n".join(visible_objects), objects_all

def is_within_observation_range(robot_pos, robot_heading, object_pos, obstacles, fov=math.pi/2, max_distance=10):
    x_robot, y_robot = robot_pos
    x_object, y_object = object_pos
    
    # Calculate the distance and direction from the robot to the object
    dx = x_object - x_robot
    dy = y_object - y_robot
    distance = math.sqrt(dx**2 + dy**2)
    
    # Check if the distance is within the maximum range
    if distance > max_distance:
        return False
    
    # Calculate the angle to the object relative to the robot
    angle_to_object = math.atan2(dy, dx)
    
    # Calculate the difference between the robot's heading and the object's angle
    angle_diff = angle_to_object - robot_heading
    
    # Normalize the angle difference to the range [-pi, pi]
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi
    
    # Check if the angle difference is within the field of view
    if abs(angle_diff) > fov / 2:
        return False
    
    # Check for obstacle occlusion
    for obstacle in obstacles:
        if line_intersects((x_robot, y_robot), (x_object, y_object), obstacle):
            return False
    
    return True

def line_intersects(p1, p2, obstacle):
    # Unpack the obstacle endpoints
    (x1, y1), (x2, y2) = obstacle
    
    # Check if two line segments (p1, p2) and (x1, y1), (x2, y2) intersect
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    def intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    
    return intersect(p1, p2, (x1, y1), (x2, y2))

def generate_executable_code(optimal_plan, env):
    # Split the plan into lines
    lines = optimal_plan.strip().split("\n")
    
    # Store the generated code
    executable_code = []

    # Iterate over each line
    for line in lines:
        # Trim whitespace
        line = line.strip()
        
        # Skip empty or invalid lines
        if not line or "Step" not in line:
            continue
        
        # Extract action and parameters
        parts = line.split(":")[1].strip().split("(")
        action = parts[0].strip()
        
        # Generate code based on the action
        if action == "push_to":
            params = parts[1].rstrip(")").split(", ")
            # executable_code.append('env0 = OriginalHighEnvWrapper(env)')
            # executable_code.append('env0 = RslRlVecEnvWrapper(env0)')
            executable_code.append(f'time_out, task_finished = push_to(env0, "{params[0]}", {params[1]}, policies, final_goal, time_step)')
        elif action == "walk_to":
            params = parts[1].rstrip(")")
            # executable_code.append('env0 = OriginalHighEnvWrapper(env)')
            # executable_code.append('env0 = RslRlVecEnvWrapper(env0)')
            executable_code.append(f'time_out, task_finished = walk_to(env0, {params}, policies, final_goal, time_step)')
        elif action == "navigate_to":
            params = parts[1].rstrip(")")
            # executable_code.append('env0 = OriginalHighEnvWrapper(env)')
            # executable_code.append('env0 = RslRlVecEnvWrapper(env0)')
            executable_code.append(f'time_out, task_finished = navigate_to(env0, {params}, policies, final_goal, time_step)')
        elif action == "climb_to":
            # with torch.inference_mode():
            #     env0 = ClimbEnvWrapper(env)
            #     env0 = RslRlVecEnvWrapper(env0)
            params = parts[1].rstrip(")")
            # executable_code.append('env0 = OriginalLowEnvWrapper(env)')
            # executable_code.append('env0 = RslRlVecEnvWrapper(env0)')
            executable_code.append(f'time_out, task_finished = climb_to(env0, {params}, policies, final_goal, time_step)')

    # Return the generated code
    return "\n".join(executable_code)


def push_to(env0, object, target_position, policies, final_goal, time_step, policy_name = "push"):
    obs, _ = env0.get_observations()
    policy = policies[policy_name]
    time_out = False
    task_finished = False
    command: CommandTerm = env.command_manager.get_term("pose_command")
    time_left = command.time_left
    # run everything in inference mode
    with torch.inference_mode():
        if time_step % 10 == 0:
            print(f"Pushing {object} to {target_position}")

            time_left = command.time_left
            print(command.time_left)
        
        robot_pos = env.unwrapped.scene["robot"].data.root_state_w
        # print("robot_pos: ", robot_pos)
        robot_heading = euler_xyz_from_quat(robot_pos[:, 3:7])[2]
        # robot_heading = env.unwrapped.scene["robot"].data.heading_w
        object_pos = env.unwrapped.scene[object].data.root_state_w
        # print("object_pos: ", object_pos[:, :2])
        object_pos_rel = object_pos[:, :2] - robot_pos[:, :2]
        # object_heading = env.unwrapped.scene["box_1"].data.heading_w

        # print("robot_pos: ", robot_pos)
        # print("robot_heading: ", robot_heading)
        # print("object_pos: ", object_pos)  
        # print("object_heading: ", object_heading)  

        # print("event_cfg: ", env.unwrapped.cfg.events)

        # obs_push = obs[:, 0:13]
        objects = env.unwrapped.scene.rigid_objects
        object_size = objects[object].cfg.spawn.size 
        # print("object_size: ", object_size)
        obs_push = torch.cat((obs[:, :13], obs[:, 17:20]), dim=1)
        
        # marker_position = torch.tensor([[target_position[0], target_position[1], 0],[final_goal[0], final_goal[1], final_goal[2]]], device=env.unwrapped.device)
        marker_position = torch.tensor([[50, 50, 0],[final_goal[0], final_goal[1], final_goal[2]]], device=env.unwrapped.device)
        my_visualizer.visualize(marker_position, torch.tensor([[0, 0, 0, 1]], device=env.unwrapped.device), marker_indices=torch.tensor([1,2]))

        # artificial command
        obs_pos_command = obs_push[:, 6:8]
        obs_push[:, 6:8] = target_position - object_pos[:, :2]
        if time_step % 10 == 0:
            print("obs_pos_command: ", obs_pos_command)
        obs_ang_command = obs_push[:, 9]
        command_mask = torch.logical_or(torch.norm(obs_pos_command, dim=-1) > 0.5, torch.abs(obs_ang_command) > 1.0)
        # command_mask = torch.norm(obs_pos_command, dim=-1) > 0.8

        obs_push[:, 3:5] = object_pos[:, :2] - robot_pos[:, :2]
        object_angle_to_robot = torch.atan2(object_pos_rel[:, 1], object_pos_rel[:, 0])
        object_angle_rel = (object_angle_to_robot - robot_heading).unsqueeze(1) % (2 * torch.pi) 
        object_angle_rel -= 2 * torch.pi * (object_angle_rel > torch.pi)
        obs_push[:, 5] = object_angle_rel
        # obs_push[:, 6:10] = object_angle_rel
        object_size_tensor = torch.tensor(object_size, device=env.unwrapped.device)
        obs_push[:, 13:16] = object_size_tensor
        # print(obs_push)
        
        # print("obs_pos_command: ", obs_pos_command)
        # print("obs_ang_command: ", obs_ang_command)
        # print("command_mask: ", command_mask)

        # Check the condition to break the loop
        if command_mask[0] == 0 and torch.norm(obs_pos_command[0]) != 0:
            task_finished = True

        # print(env.unwrapped.cfg.scene.camera)
        # print(env.scene)
        # camera_cfg = env.unwrapped.scene.camera
        # camera = Camera(camera_cfg)
    
        # actions = torch.where(command_mask[:, None], policy(obs_push), torch.zeros_like(policy(obs_push)))
        # actions = torch.where(command_mask[:, None], policy(obs_push), policy2(obs_walk))
        policy_start_time = time.time()
        actions = policy(obs_push)
        policy_end_time = time.time()
        # print("policy time: ", policy_end_time - policy_start_time)

        # stay static
        # actions = torch.zeros_like(actions)
        
        obs, _, _, _ = env0.step(actions)

        if time_left <= 0.3:
            # print("time out")
            time_out = True
            # break

    return time_out, task_finished

def walk_to(env0, target_position, policies, final_goal, time_step, policy_name = "walk"):
    obs, _ = env0.get_observations()
    policy = policies[policy_name]
    time_out = False
    task_finished = False
    command: CommandTerm = env.command_manager.get_term("pose_command")
    time_left = command.time_left
    # run everything in inference mode
    with torch.inference_mode():
        if time_step % 10 == 0:
            print(f"Walking to {target_position}")

            time_left = command.time_left
            print(command.time_left)

        robot_pos = env.unwrapped.scene["robot"].data.root_pos_w
        # print("robot_pos: ", robot_pos)
        robot_state = env.unwrapped.scene["robot"].data.root_state_w
        _, _, robot_yaw = euler_xyz_from_quat(env.unwrapped.scene["robot"].data.root_quat_w)
        yaw_err = torch.abs(robot_yaw - target_position[2])

        goal, _ = subtract_frame_transforms(robot_state[:, :3], robot_state[:, 3:7], target_position)
        
        robot_pos_command = obs[:, 13:15]
        robot_ang_command = obs[:, 16]
        # command_mask = torch.logical_or(torch.norm(robot_pos_command, dim=-1) > 0.5, torch.abs(robot_ang_command) > 0.3)
        command_mask = torch.logical_or(torch.norm(goal[:, :2], dim=-1) > 0.5, yaw_err > 0.3)
        # command_mask = torch.norm(goal[:, :2], dim=-1) > 0.5
        
        # print("obs_pos_command: ", robot_pos_command)
        # print("obs_ang_command: ", obs_ang_command)
        # print("command_mask: ", command_mask)
        # Check the condition to break the loop
        if command_mask[0] == 0: # and torch.norm(robot_pos_command[0]) != 0:
            task_finished = True

        obs_walk = torch.cat((obs[:, :3], obs[:, 10:17]), dim=1)

        # marker_position = torch.tensor([[target_position[0], target_position[1], 0.5],[final_goal[0], final_goal[1], final_goal[2]]], device=env.unwrapped.device)
        marker_position = torch.tensor([[50, 50, 0],[final_goal[0], final_goal[1], final_goal[2]]], device=env.unwrapped.device)
        my_visualizer.visualize(marker_position, torch.tensor([[0, 0, 0, 1]], device=env.unwrapped.device), marker_indices=torch.tensor([3,2]))

        # artificial command
        obs_walk[:, 6:8] = goal[:, :2]
        obs_walk[:, 9] = 1.5
        # print("goal: ", goal)

        # print("obs_walk: ", obs_walk)

        actions = policy(obs_walk)
        # actions = torch.where(command_mask[:, None], policy(obs_push), policy2(obs_walk))
        # actions = policy2(obs_walk)

        # stay static
        # actions = torch.zeros_like(actions)
        
        obs, _, _, _ = env0.step(actions)

        if time_left <= 0.3:
            # print("time out")
            time_out = True

    return time_out, task_finished

def navigate_to(env0, target_position, policies, final_goal, time_step, policy_name = "navigation"):
    obs, _ = env0.get_observations()
    policy = policies[policy_name]
    time_out = False
    task_finished = False
    command: CommandTerm = env.command_manager.get_term("pose_command")
    time_left = command.time_left
    # run everything in inference mode
    with torch.inference_mode():
        policy_start_time = time.time()

        if time_step % 10 == 0:
            print(f"Navigation to {target_position}")

            time_left = command.time_left
            print(command.time_left)

        robot_pos = env.unwrapped.scene["robot"].data.root_pos_w
        # print("robot_pos: ", robot_pos)
        robot_state = env.unwrapped.scene["robot"].data.root_state_w

        # artificial command
        goal, _ = subtract_frame_transforms(robot_state[:, :3], robot_state[:, 3:7], target_position)
        # print("goal: ", goal[:, :2])

        # marker_position = torch.tensor([[target_position[0], target_position[1], 0.5],[final_goal[0], final_goal[1], final_goal[2]]], device=env.unwrapped.device)
        marker_position = torch.tensor([[50, 50, 0],[final_goal[0], final_goal[1], final_goal[2]]], device=env.unwrapped.device)
        my_visualizer.visualize(marker_position, torch.tensor([[0, 0, 0, 1]], device=env.unwrapped.device), marker_indices=torch.tensor([3,2]))
    
        # print("obs_walk: ", obs_walk)
        
        command_mask = torch.norm(goal[:, :2], dim=-1) > 0.6
        # print("norm: ", torch.norm(goal, dim=1))   
        # command_mask = torch.norm(obs_pos_command, dim=-1) > 0.8
        
        # print("command_mask: ", command_mask)
        # Check the condition to break the loop
        if command_mask[0] == 0: # and torch.norm(robot_pos_command[0]) != 0:
            task_finished = True

        # obs_walk = torch.cat((obs[:, :3], obs[:, 10:17]), dim=1)
        obs_walk = torch.cat((obs[:, :3], obs[:, 10:17], obs[:, 59:246]), dim=1)
        # obs_walk = torch.cat((obs[:, :3], obs[:, 10:17], obs[:, 50:246]), dim=1)
        # print("obs shape: ", obs.shape)
        # print("obs_walk shape: ", obs_walk.shape)
        obs_walk[:, 6:8] = goal[:, :2]

        actions = policy(obs_walk)

        policy_end_time = time.time()
        # print("policy time: ", policy_end_time - policy_start_time)

        # stay static
        # actions = torch.zeros_like(actions)
        
        obs, _, _, _ = env0.step(actions)

        if time_left <= 0.3:
            # print("time out")
            time_out = True

    return time_out, task_finished


def climb_to(env0, target_position, policies, final_goal, time_step, policy_name = "climb"):
    obs, _ = env0.get_observations()
    policy = policies[policy_name]
    time_out = False
    task_finished = False
    command: CommandTerm = env.command_manager.get_term("pose_command")
    time_left = command.time_left
    # run everything in inference mode
    with torch.inference_mode():
        if time_step % 10 == 0:

            print(f"Climbing to {target_position}")

            time_left = command.time_left
            print(command.time_left)

        # marker_position = torch.tensor([[target_position[0], target_position[1], target_position[2]],[final_goal[0], final_goal[1], final_goal[2]]], device=env.unwrapped.device)
        # my_visualizer.visualize(marker_position, torch.tensor([[0, 0, 0, 1]], device=env.unwrapped.device), marker_indices=torch.tensor([3,2]))
        
        robot_pos = env.unwrapped.scene["robot"].data.root_pos_w
        # print("robot_pos: ", robot_pos)
        robot_state = env.unwrapped.scene["robot"].data.root_state_w
        
        goal, _ = subtract_frame_transforms(robot_state[:, :3], robot_state[:, 3:7], target_position)
        # print("goal: ", goal[:, :2])
        command_mask = torch.norm(goal[:, :2], dim=-1) > 0.8
        # print("command_mask: ", command_mask)
        # Check the condition to break the loop
        if command_mask[0] == 0: # and torch.norm(robot_pos_command[0]) != 0:
            task_finished = True

        obs_walk = torch.cat((obs[:, :3], obs[:, 20:23], obs[:, 10:17], obs[:, 23:246]), dim=1)
        # print("obs shape: ", obs.shape)
        # print("obs_walk shape: ", obs_walk.shape)

        # artificial command
        obs_walk[:, 9:11] = goal[:, :2]
        obs_walk[:, 12] = 0
        # obs_walk[:, 12] = 1.5

        # print("obs_walk: ", obs_walk)

        actions = policy(obs_walk)
        # actions = torch.where(command_mask[:, None], policy(obs_push), policy2(obs_walk))
        # actions = policy2(obs_walk)

        # stay static
        # actions = torch.zeros_like(actions)
        
        obs, _, _, _ = env0.step(actions)

        if time_left <= 0.3:
            # print("time out")
            time_out = True

    return time_out, task_finished

def update_grid_map(grid_map, lidar_data, resolution=0.1):
    lidar_data = lidar_data.ray_hits_w
    lidar_data = lidar_data.cpu().numpy()
    
    x = lidar_data[0, :, 0]
    y = lidar_data[0, :, 1]
    z = lidar_data[0, :, 2]
    
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    
    grid_width = int((x_max - x_min) / resolution) + 1
    grid_height = int((y_max - y_min) / resolution) + 1
    
    for xi, yi, zi in zip(x, y, z):
        grid_x = int((xi - x_min) / resolution)
        grid_y = int((yi - y_min) / resolution)
        
        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
            grid_map[grid_y, grid_x] = zi
    
    return grid_map

def inflate_obstacles(grid_map, inflation_radius):
    structuring_element = np.ones((2 * inflation_radius + 1, 2 * inflation_radius + 1))
    
    obstacles = ~np.isnan(grid_map) & (grid_map > 0.1)
    
    inflated_obstacles = binary_dilation(obstacles, structure=structuring_element)
    
    inflated_grid_map = np.where(inflated_obstacles, 1, grid_map)
    
    return inflated_grid_map

def animate(grid_map, x_min, x_max, y_min, y_max, robot_pos, robot_yaw, final_goal, path, mid_point):
    plt.clf()

    # inflation_radius = 1
    # grid_map = inflate_obstacles(grid_map, inflation_radius)
    
    # Plot the map
    plt.imshow(grid_map, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='viridis_r')
    plt.colorbar(label='Height')
    plt.title('Height Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Plot robot position
    robot_pos = robot_pos.cpu().numpy()
    # print("robot_pos: ", robot_pos)
    plt.plot(robot_pos[0][0], robot_pos[0][1], 'ro', label='Robot')  # Red dot for robot position
    
    # Plot robot orientation
    robot_yaw = robot_yaw.cpu().item()
    yaw_length = 0.5  # Length of the orientation arrow
    arrow_dx = yaw_length * np.cos(robot_yaw)
    arrow_dy = yaw_length * np.sin(robot_yaw)
    plt.arrow(robot_pos[0][0], robot_pos[0][1], arrow_dx, arrow_dy, head_width=0.2, head_length=0.2, fc='r', ec='r')
    
    # Plot goal position
    goal = final_goal.cpu().numpy()
    # print("goal: ", goal)
    plt.plot(goal[0], goal[1], 'gx', label='Goal')  # Green X for goal position

    if path:
        path_x, path_y = zip(*path)
        plt.plot(path_x, path_y, 'b-', label='Path')  # Blue line for path

        # print("mid_point: ", mid_point)
        mid_point = mid_point.cpu().numpy()
        plt.plot(mid_point[0], mid_point[1], 'bo', label='Mid Point')  # Blue dot for mid point
    
    plt.legend()
    plt.show(block=False)
    plt.pause(0.01)

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))/10

def path_finding(grid_map, robot_pos, final_goal):
    map_grid_size = 0.1
    grid_map = np.nan_to_num(grid_map, nan=0)
    inflation_radius = 3
    grid_map = inflate_obstacles(grid_map, inflation_radius)
    # print("grid_map: ", grid_map)
    rows, cols = grid_map.shape
    start = tuple(map(int, (robot_pos[0, :2]/map_grid_size).cpu().numpy()))
    goal = tuple(map(int, (final_goal[:2]/map_grid_size).cpu().numpy()))
    # print("start: ", start)
    # print("goal: ", goal)

    goal_index = (goal[0] + 50, goal[1] + 50)
    if grid_map[goal_index[1],goal_index[0]] > 0.1:
        return [], 0

    start_index = (start[0] + 50, start[1] + 50)
    if grid_map[start_index[1],start_index[0]] > 0.1:
        return [], 2  
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    closed_set = []
    
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    step_size = 1
    next_step = [(-step_size, 0), (step_size, 0), (0, -step_size), (0, step_size), 
                 (-step_size, -step_size), (-step_size, step_size), 
                 (step_size, -step_size), (step_size, step_size)]
    
    while open_set:
        _, current = heapq.heappop(open_set)
        # print("current: ", current)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
                # print("current1: ", current)
            path.append(start)
            path.reverse()
            return path, 1
        
        for dx, dy in next_step:
            neighbor = (current[0] + dx, current[1] + dy)
            map_index = (current[0] + dx + 50, current[1] + dy + 50)

            if neighbor in closed_set:
                continue
            
            if -40 <= neighbor[0] < 40 and -40 <= neighbor[1] < 40 and grid_map[map_index[1], map_index[0]] < 0.1:
                tentative_g_score = g_score[current] + np.sqrt(dx**2 + dy**2) + grid_map[map_index[1], map_index[0]]*10
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # print("neighbor: ", neighbor)
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
    return [], 0  # No path found

def move(env0, env, policies, final_goal, start_time):
    time_step = 0
    traditional_navigation = False
    task_finished = False
    time_out = False
    id = 2
    last_skill = ""
    local_vars = {}
    objects_all = {}

    # map initialization
    resolution = 0.1
    x_min, x_max = -5, 5
    y_min, y_max = -5, 5
    grid_width = int((x_max - x_min) / resolution) + 1
    grid_height = int((y_max - y_min) / resolution) + 1

    grid_map = np.full((grid_height, grid_width), np.nan)

    path = []
    mid_point = []
    flag = 0
    first_time = True

    initial_plan = True
    replan = False
    LLM_planning = False

    retry_count = 0

    planning_time = 0
    execution_time = 0

    while simulation_app.is_running():
        planning_time_start = time.time()
        # get observation
        if time_step % 10 == 0:
            # lidar data
            lidar_data = env.unwrapped.scene.sensors["height_scanner2"].data
            # object_lidar_data = env.unwrapped.scene.sensors["object_height_scanner"].data
            # print("object_lidar_data: ", object_lidar_data)

            # image data
            rgb_api = get_viewport_from_window_name("RGB")
            capture_viewport_to_file(rgb_api, r"image/rgb_{}.png".format(time_step))
            thirdview_api = get_viewport_from_window_name("Viewport")
            capture_viewport_to_file(thirdview_api, r"image/thirdview_{}.png".format(time_step))
            thirdview_api2 = get_viewport_from_window_name("robot-fixed view")
            capture_viewport_to_file(thirdview_api2, r"image/fixed_view_{}.png".format(time_step))

            _, _, robot_yaw = euler_xyz_from_quat(env.unwrapped.scene["robot"].data.root_quat_w)
            robot_pos = env.unwrapped.scene["robot"].data.root_pos_w

            grid_map = update_grid_map(grid_map, lidar_data, 0.1)

            # if time_step % 20 == 0:

            #     path, flag = path_finding(grid_map, robot_pos, final_goal)
            #     path = [(x*0.1, y*0.1) for x, y in path]
            #     if flag == 1:
            #         # print("path: ", path)
            #         print("Path found")
            #         traditional_navigation = True
            #         if len(path) > 20:
            #             # mid_point = path[10]
            #             mid_point = torch.tensor([path[20][0], path[20][1], 0], device=env.unwrapped.device)
            #         else:
            #             # mid_point = path[-1]
            #             mid_point = torch.tensor([path[-1][0], path[-1][1], 0], device=env.unwrapped.device)
            #     elif flag == 2:
            #         print("start point is not valid, using previous mid point")
            #         traditional_navigation = True
            #         mid_point = final_goal
            #     else:
            #         print("No path found")
            #         traditional_navigation = False

            # plot
            # if time_step % 20 == 0:
            #     animate(grid_map, x_min, x_max, y_min, y_max, robot_pos, robot_yaw, final_goal, path, mid_point)
    

        if time_step % 50 == 0:
            current_object, objects_all, object_description = get_observation(env, objects_all)
            # current_keys = set(objects_all.keys())
            # new_keys = current_keys - previous_keys
            # if new_keys:
            #     replan = cognition_replanning

            # previous_object = objects_all
            # previous_keys = set(previous_object.keys())

            if time_step > 0 & LLM_planning == False:
                img_path = r"image/rgb_{}.png".format(time_step-10)
                # image = Image.open(img_path)
                # plt.imshow(image)
                # plt.show()

                # inference_start_time = time.time()
                # print("Inference time: ", time.time() - inference_start_time)

                # idx = clip(img_path, processor_clip, model_clip)
                # idx = blip(img_path, processor_blip, model_blip)

                # if idx == 0:
                #     traditional_navigation = True
                # elif idx == 1:
                #     traditional_navigation = False

                # if LLM_planning:
                #     traditional_navigation = False
            
        close_to_platform = torch.tensor([2.1, 1.7], device=env.unwrapped.device)
        close_to_platform1 = torch.tensor([2.1, 1.7], device=env.unwrapped.device)
        close_to_platform2 = torch.tensor([2.1, 1.7], device=env.unwrapped.device)
        close_to_box_1 = torch.tensor([2.1, 0.0, 0.0], device=env.unwrapped.device)
        position_2 = torch.tensor([-2.0, -4.0], device=env.unwrapped.device)
        near_platform = torch.tensor([-2.0, -1.3, 0], device=env.unwrapped.device)
        # mid_point = torch.tensor([0, 0, 0], device=env.unwrapped.device)
        middle_point = torch.tensor([1.5, 1.0, 0], device=env.unwrapped.device)
        middle_point2 = torch.tensor([1.5, 2.8], device=env.unwrapped.device)
        middle_point3 = torch.tensor([1.0, -2.5, 0], device=env.unwrapped.device)
        top_of_platform = torch.tensor([2.1, 2.5, 0.4], device=env.unwrapped.device)
        top_of_box_1 = torch.tensor([-5.5, -1.3, 0.4], device=env.unwrapped.device)
            
        # select skill based on observation
        if traditional_navigation:
            # mid_point = torch.tensor([mid_point[0], mid_point[1], 0], device=env.unwrapped.device)
            optimal_plan = """
            Step 0: start
            Step 1: navigate_to(mid_point)
            """
            # optimal_plan = """
            # Step 0: start
            # Step 1: navigate_to(final_goal)
            # """

            if first_time:
                with torch.inference_mode():
                    env0 = OriginalHighEnvWrapper(env)
                    env0 = RslRlVecEnvWrapper(env0)
                first_time = False
        else:
            # -------------LLM reasoning-----------------
            # LLM_planning = True
            # if initial_plan or replan:
            #     optimal_plan_all = LLM_task_planning(object_description, final_goal)
            #     optimal_plan_all = '\n' + optimal_plan_all + '\n'
            #     initial_plan = False
            #     if replan:
            #         replan = False

            #     print("optimal_plan: ", optimal_plan_all)

            # traditional_navigation = True

            # simulation 1

            # optimal_plan_all = """
            # Step 0: start
            # Step 1: push_to(box_1, close_to_platform)
            # Step 2: navigate_to(middle_point)
            # Step 1: push_to(box_2, middle_point2)
            # Step 2: navigate_to(final_goal)
            # """

            # optimal_plan_all = """
            # # Step 0: start
            # # Step 1: push_to(box_1, close_to_platform)
            # # Step 2: navigate_to(middle_point)
            # # Step 2: navigate_to(final_goal)
            # # """

            # optimal_plan_all = """
            # Step 0: start
            # Step 1: push_to(box_1, close_to_platform)
            # Step 2: navigate_to(position_2)
            # Step 2: navigate_to(middle_point)
            # Step 2: navigate_to(top_of_box_1)
            # Step 2: navigate_to(close_to_box_1)
            # Step 2: navigate_to(final_goal)
            # """

            # close_to_platform = torch.tensor([0.5, -3.2], device=env.unwrapped.device)
            # position_2 = torch.tensor([-0.4, -1.9, 0], device=env.unwrapped.device)
            # near_platform = torch.tensor([-2.0, -1.3, 0], device=env.unwrapped.device)
            # middle_point = torch.tensor([1.9, 1.7, 0], device=env.unwrapped.device)
            # middle_point2 = torch.tensor([1.9, 2.8], device=env.unwrapped.device)
            # close_to_box_1 = torch.tensor([-0.2, 3.0, 0.0], device=env.unwrapped.device)
            # top_of_platform = torch.tensor([-5.5, -1.3, 0.4], device=env.unwrapped.device)
            # top_of_box_1 = torch.tensor([-0.2, 1.0, 0.4], device=env.unwrapped.device)

            # simulation 2

            # optimal_plan_all =  """
            # Step 0: start
            # Step 1: navigate_to(close_to_box_1)
            # Step 3: walk_to(top_of_box_1)
            # Step 4: walk_to(final_goal)
            # """

            optimal_plan_all =  """
            Step 0: start
            Step 1: navigate_to(close_to_box_1)
            Step 2: push_to(box_1, close_to_platform)
            Step 3: climb_to(top_of_box_1)
            Step 4: walk_to(final_goal)
            """

            # optimal_plan_all =  """
            # Step 0: start
            # Step 1: navigate_to(middle_point)
            # Step 1: walk_to(near_platform)
            # Step 1: navigate_to(close_to_box_1)
            # Step 2: push_to(box_1, close_to_platform)
            # Step 3: walk_to(top_of_box_1)
            # Step 4: walk_to(final_goal)
            # """

            close_to_box_1 = torch.tensor([2.7, 0.0, 0.0], device=env.unwrapped.device)
            close_to_platform = torch.tensor([2.7, 2.0], device=env.unwrapped.device)
            top_of_box_1 = torch.tensor([3.0, 3.5, 0.4], device=env.unwrapped.device)
            top_of_platform = torch.tensor([3.0, 3.0, 0.4], device=env.unwrapped.device)
            position_2 = torch.tensor([-1, -2.5, 0], device=env.unwrapped.device)
            near_platform = torch.tensor([0.5, 0.0, 0.0], device=env.unwrapped.device)
            middle_point = torch.tensor([0.7, 2.7, 0], device=env.unwrapped.device)
            middle_point2 = torch.tensor([1.4, 2.5], device=env.unwrapped.device)

            # simulation 3

            # optimal_plan_all =  """
            # Step 0: start
            # Step 1: navigate_to(close_to_box_1)
            # Step 3: walk_to(top_of_box_1)
            # Step 4: walk_to(final_goal)
            # """

            # optimal_plan_all =  """
            # Step 0: start
            # Step 1: navigate_to(close_to_box_1)
            # Step 2: push_to(box_1, close_to_platform1)
            # Step 3: navigate_to(position_2)
            # Step 4: walk_to(top_of_box_1)
            # Step 4: walk_to(final_goal)
            # """

            # high difficulty
            # optimal_plan_all =  """
            # Step 0: start
            # Step 1: navigate_to(middle_point)
            # Step 1: walk_to(near_platform)
            # Step 2: push_to(box_2, close_to_platform2)
            # Step 3: navigate_to(top_of_platform)
            # Step 3: walk_to(position_2)
            # Step 3: climb_to(top_of_box_1)
            # Step 4: walk_to(final_goal)
            # """

            # # Step 2: push_to(box_1, close_to_platform1)

            # close_to_box_1 = torch.tensor([0.2, 1.0, 0.0], device=env.unwrapped.device)
            # close_to_platform1 = torch.tensor([1.8, 1.8], device=env.unwrapped.device)
            # top_of_box_1 = torch.tensor([2.2, 3.0, 1.5], device=env.unwrapped.device)
            # top_of_platform = torch.tensor([2.3, -0.5, 0], device=env.unwrapped.device)
            # position_2 = torch.tensor([2.2, 0.0, 1.57], device=env.unwrapped.device)
            # near_platform = torch.tensor([2.8, -0.5, 1.5], device=env.unwrapped.device)
            # middle_point = torch.tensor([1.8, 0.0, 0], device=env.unwrapped.device)
            # middle_point2 = torch.tensor([-0.8, -0.5, 0], device=env.unwrapped.device)

            # # for high difficulty
            # # close_to_platform2 = torch.tensor([1.6, 1.6], device=env.unwrapped.device)
            # close_to_platform2 = torch.tensor([1.6, 0.3], device=env.unwrapped.device)
            # top_of_box_1 = torch.tensor([1.9, 3.0, 0.4], device=env.unwrapped.device)
            # position_2 = torch.tensor([1.8, -0.5, 0], device=env.unwrapped.device)

            # simulation 4

            # optimal_plan_all =  """
            # Step 0: start
            # Step 1: navigate_to(close_to_box_1)
            # Step 3: walk_to(top_of_box_1)
            # Step 4: walk_to(final_goal)
            # """

            # optimal_plan_all =  """
            # Step 0: start
            # Step 1: navigate_to(close_to_box_1)
            # Step 2: push_to(box_1, close_to_platform)
            # Step 3: navigate_to(position_2)
            # Step 4: walk_to(top_of_box_1)
            # Step 4: walk_to(final_goal)
            # """

            # optimal_plan_all =  """
            # Step 0: start
            # Step 1: navigate_to(close_to_box_1)
            # Step 2: push_to(box_1, close_to_platform)
            # Step 3: walk_to(middle_point)
            # Step 1: climb_to(middle_point2)
            # Step 3: walk_to(position_2)
            # Step 3: climb_to(middle_point3)
            # Step 3: walk_to(final_goal)
            # """

            # close_to_box_1 = torch.tensor([-2.5, 0.0, -0.8], device=env.unwrapped.device)
            # close_to_platform = torch.tensor([-0.4, 0.4], device=env.unwrapped.device)
            # top_of_box_1 = torch.tensor([2.2, 3.0, 0.4], device=env.unwrapped.device)
            # top_of_platform = torch.tensor([3.0, 3.0, 0.4], device=env.unwrapped.device)
            # position_2 = torch.tensor([2.5, -0.2, 1.0], device=env.unwrapped.device)
            # near_platform = torch.tensor([1.0, 0.0, 0.0], device=env.unwrapped.device)
            # middle_point = torch.tensor([-2.2, 0.5, 0.0], device=env.unwrapped.device)
            # middle_point2 = torch.tensor([2.0, 0.4, 0], device=env.unwrapped.device)
            # middle_point3 = torch.tensor([2.5, 3.5, 0], device=env.unwrapped.device)

            # close_to_platform = torch.tensor([1.6, 1.6], device=env.unwrapped.device)
            # top_of_box_1 = torch.tensor([1.9, 3.0, 0.4], device=env.unwrapped.device)
            # position_2 = torch.tensor([1.8, 0.0, 0], device=env.unwrapped.device)

            # simulation 5

            # optimal_plan_all =  """
            # Step 0: start
            # Step 1: navigate_to(close_to_box_1)
            # Step 1: walk_to(close_to_box_1)
            # Step 3: walk_to(middle_point)
            # Step 3: walk_to(position_2)
            # Step 3: walk_to(middle_point2)
            # Step 3: push_to(box_2, close_to_platform)
            # Step 3: climb_to(middle_point3)
            # Step 3: walk_to(final_goal)
            # """

            # optimal_plan_all =  """
            # Step 0: start
            # Step 3: walk_to(middle_point2)
            # Step 3: climb_to(middle_point3)
            # Step 3: walk_to(final_goal)
            # """

            # close_to_box_1 = torch.tensor([-1.5, 1.0, 0.0], device=env.unwrapped.device)
            # close_to_platform = torch.tensor([2.5, 1.3], device=env.unwrapped.device)
            # top_of_box_1 = torch.tensor([2.2, 3.0, 0.4], device=env.unwrapped.device)
            # top_of_platform = torch.tensor([3.0, 3.0, 0.4], device=env.unwrapped.device)
            # position_2 = torch.tensor([1.0, -0.5, 0], device=env.unwrapped.device)
            # near_platform = torch.tensor([1.0, 0.0, 0.0], device=env.unwrapped.device)
            # middle_point = torch.tensor([0.5, 1.0, 0.5], device=env.unwrapped.device)
            # middle_point2 = torch.tensor([2.5, -0.5, 1.5], device=env.unwrapped.device)
            # middle_point3 = torch.tensor([1.0, -0.5, 0], device=env.unwrapped.device)
            # middle_point3 = torch.tensor([2.5, 3.5, 0], device=env.unwrapped.device)
            

            # print(len(optimal_plan_all.splitlines()))

            # skill selection
            if time_out:
                retry_count += 1
                print("current task is time out, retrying to complete the task")
                if retry_count > 2:
                    print("retry count exceed, failed to complete the task")
                    # id = id + 1
                    replan = True
                    retry_count = 0
            if task_finished:
                print("current task is finished, start to execute the next task")
                id = id + 1
                
            id = min(id, len(optimal_plan_all.splitlines())-2)
            optimal_plan = optimal_plan_all.splitlines()[id]
            # print("optimal_plan: ", optimal_plan)

            line = optimal_plan.strip()
            
            parts = line.split(":")[1].strip().split("(")
            skill = parts[0].strip()

            if skill != last_skill:
                with torch.inference_mode():
                    if skill == "climb_to":
                        env0 = OriginalLowEnvWrapper(env)
                        env0 = RslRlVecEnvWrapper(env0)
                    else:
                        env0 = OriginalHighEnvWrapper(env)
                        env0 = RslRlVecEnvWrapper(env0)

            last_skill = skill

        planning_time_end = time.time()
        planning_time = planning_time + (planning_time_end - planning_time_start)

        execution_time_start = time.time()

        # execute the selected skill
        executable_code = generate_executable_code(optimal_plan, env)
        # print("execute code: ", executable_code)

        local_vars = {
            'push_to': push_to,
            'navigate_to': navigate_to,
            'walk_to': walk_to,
            'climb_to': climb_to,
            'time_out': time_out,
            'task_finished': task_finished,
            'mid_point': mid_point,
            'env0': env0,
            'policies': policies,
            'final_goal': final_goal,
            'time_step': time_step,
            'close_to_platform': close_to_platform,
            'close_to_platform1': close_to_platform1,
            'close_to_platform2': close_to_platform2,
            'close_to_box_1': close_to_box_1,
            'position_2': position_2,
            'near_platform': near_platform,
            'top_of_platform': top_of_platform,
            'top_of_box_1': top_of_box_1,
            'middle_point': middle_point,
            'middle_point2': middle_point2,
            'middle_point3': middle_point3,
        }

        with torch.inference_mode():
            exec(executable_code, {}, local_vars)
            time_out = local_vars["time_out"]
            task_finished = local_vars["task_finished"]

        robot_state = env.unwrapped.scene["robot"].data.root_state_w

        goal, _ = subtract_frame_transforms(robot_state[:, :3], robot_state[:, 3:7], final_goal)
        # print("goal: ", goal[:, :2])
        
        command_mask = torch.norm(goal[:, :2], dim=-1) > 0.5

        execution_time_end = time.time()
        execution_time = execution_time + (execution_time_end - execution_time_start)

        end_time = time.time()
        simulation_time = end_time - start_time
        
        if simulation_time > 120:
            print("simulation ended")
            # return False
            break

        if time_step % 10 == 0:
            print("distance to goal: ", torch.norm(goal, dim=-1))   
            print("time_spent: ", time_step)
            print("planning time: ", planning_time)
            print("execution time: ", execution_time)
            print("simulation time: ", simulation_time)

        if command_mask[0] == 0: # and torch.norm(robot_pos_command[0]) != 0:
            print("goal reached")
            return True, planning_time, execution_time
            break

        # if time_step > 1000:
        #     print("simulation ended")
        #     break

        time_step += 1
        # execution_time = time_step * 0.02

    return False, planning_time, execution_time


def get_observation(env, objects_all):
    print("---------------------------------")
    current_object, objects_all = get_object_description(env, objects_all)
    print("Current visible objects:")
    print(current_object)

    object_description = "\n".join(objects_all.values())
    platform_description = "* [platform]: <center>: [2.0, 2.5, 0.0]; <size>: [3.0, 1.0, 0.4]. <movable>: False; <on_top_of>: ground"
    object_description = object_description + "\n" + platform_description
    print("Objects description:")
    print(object_description)

    return current_object, objects_all, object_description


if __name__ == "__main__":
    # initialization

    img_path = "./pic1.png"
    img_path2 = "./result.jpg"

    # blip(img_path)
    # clip(img_path)
    # clip(img_path2)

    env0, env, policies = init_sim()

    (test_graph, new_nodes, _, _) = og.Controller.edit(
        {"graph_path": "/ActionGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("createViewport1", "omni.isaac.core_nodes.IsaacCreateViewport"),
                ("createViewport2", "omni.isaac.core_nodes.IsaacCreateViewport"),
                ("createViewport3", "omni.isaac.core_nodes.IsaacCreateViewport"),
                ("getRenderProduct2", "omni.isaac.core_nodes.IsaacGetViewportRenderProduct"),
                ("getRenderProduct3", "omni.isaac.core_nodes.IsaacGetViewportRenderProduct"),
                ("setCamera1", "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct"),
                ("setCamera2", "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct"),
                ("setCamera3", "omni.isaac.core_nodes.IsaacSetCameraOnRenderProduct"),
            ],
            og.Controller.Keys.CONNECT: [
                ("OnTick.outputs:tick", "createViewport1.inputs:execIn"),
                ("OnTick.outputs:tick", "createViewport2.inputs:execIn"),
                ("OnTick.outputs:tick", "createViewport3.inputs:execIn"),
                ("createViewport2.outputs:execOut", "getRenderProduct2.inputs:execIn"),
                ("createViewport2.outputs:viewport", "getRenderProduct2.inputs:viewport"),
                ("getRenderProduct2.outputs:execOut", "setCamera2.inputs:execIn"),
                ("getRenderProduct2.outputs:renderProductPath", "setCamera2.inputs:renderProductPath"),
                ("createViewport3.outputs:execOut", "getRenderProduct3.inputs:execIn"),
                ("createViewport3.outputs:viewport", "getRenderProduct3.inputs:viewport"),
                ("getRenderProduct3.outputs:execOut", "setCamera3.inputs:execIn"),
                ("getRenderProduct3.outputs:renderProductPath", "setCamera3.inputs:renderProductPath"),
                
            ],
            og.Controller.Keys.SET_VALUES: [
                ("createViewport1.inputs:viewportId", 0),
                ("createViewport2.inputs:viewportId", 1),
                ("createViewport3.inputs:viewportId", 2),
                # ("createViewport1.inputs:name", "Third-person view"),
                ("createViewport2.inputs:name", "RGB"),
                ("createViewport3.inputs:name", "robot-fixed view"),
                ("setCamera2.inputs:cameraPrim", [usdrt.Sdf.Path("/World/envs/env_0/Robot/base/front_cam")]),
                # ("setCamera2.inputs:cameraPrim", [usdrt.Sdf.Path("/World/front_cam")]),
                ("setCamera3.inputs:cameraPrim", [usdrt.Sdf.Path("/World/envs/env_0/Robot/base/third_view_cam")]),
                # ("setCamera3.inputs:")
            ],
        },
    )

    simulation_app.update()

    viewports = []

    # for viewport_name in ["Viewport", "Viewport 1", "Viewport 2"]:
    #     viewport_api = get_viewport_from_window_name(viewport_name)
    #     viewports.append(viewport_api)

    main_viewport = omni.ui.Workspace.get_window("Viewport")
    rgb_viewport = omni.ui.Workspace.get_window("RGB")
    robot_viewport = omni.ui.Workspace.get_window("robot-fixed view")
    if main_viewport is not None and rgb_viewport is not None:
        rgb_viewport.dock_in(main_viewport, omni.ui.DockPosition.RIGHT, 0.5)
        robot_viewport.dock_in(main_viewport, omni.ui.DockPosition.BOTTOM, 0.5)

    final_goal = torch.tensor([3.0, 3.0, 0.0], device=env.unwrapped.device)
    final_goal = torch.tensor([3.0, 3.0, 0.6], device=env.unwrapped.device)

    # move
    simulation_time = []
    planning_time = []
    execution_time = []
    ii = 0
    while ii < 10:
        print("------------ New Simulation ----------------")
        objects_all = {}
        with torch.inference_mode():
            # reset the environment
            env0 = OriginalEnvWrapper(env)
            env0 = RslRlVecEnvWrapper(env0)

            # initialize the robot
            actions = torch.zeros((1, 3))
            num = 0
            while num < 3:
                _, _, _, _ = env0.step(actions)
                num += 1
            
        start_time = time.time()
        goal_reach, one_planning_time, one_execution_time = move(env0, env, policies, final_goal, start_time)
        end_time = time.time()
        simulation_time.append(end_time - start_time)
        print("Simulation time: ", simulation_time)
        planning_time.append(one_planning_time)
        print("Planning time: ", planning_time)
        execution_time.append(one_execution_time)
        print("Execution time: ", execution_time)

        # if goal_reach:
        #     break

        ii += 1

    time_mean = np.mean(simulation_time)
    time_std = np.std(simulation_time)
    print("Mean simulation time: ", time_mean)
    print("Std simulation time: ", time_std)

    planning_mean = np.mean(planning_time)
    planning_std = np.std(planning_time)
    print("Mean planning time: ", planning_mean)
    print("Std planning time: ", planning_std)

    execution_mean = np.mean(execution_time)
    execution_std = np.std(execution_time)
    print("Mean execution time: ", execution_mean)
    print("Std execution time: ", execution_std)

    # close the simulator
    env.close()

    # close sim app
    simulation_app.close()
