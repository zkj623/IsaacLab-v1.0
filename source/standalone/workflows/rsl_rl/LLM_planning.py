# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""


import os
import torch
import numpy as np
import heapq
import copy
import carb

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

from PIL import Image
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration

from transformers import CLIPProcessor, CLIPModel

model = YOLO('yolov8m.pt')

torch.set_printoptions(sci_mode=False, precision=4)

API_SECRET_KEY = "sk-zk2a34977de3d68b2fb0058cfe0b531907e66b5db20733c7"
BASE_URL = "https://api.zhizengzeng.com/v1/"
time_left = 15

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


def blip(img_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

    # processor = BlipProcessor.from_pretrained("./pretrained_models/blip-image-captioning-base")
    # model = BlipForConditionalGeneration.from_pretrained("./pretrained_models/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

    raw_image = Image.open(img_path).convert('RGB')

    # conditional image captioning
    # text = "You are a quadruped robot. Help me describe what you see in the picture."
    text = "box"
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    # >>> a photography of a woman and her dog

    # unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))

def clip(img_path):
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    # processor = CLIPProcessor.from_pretrained("./pretrained_models/clip-vit-base-patch32")
    # model = CLIPModel.from_pretrained("./pretrained_models/clip-vit-base-patch32", torch_dtype=torch.float16).to("cuda")
    raw_image = Image.open(img_path).convert('RGB')

    text=["the road is clear", "the road is blocked by boxes"]
    inputs = processor(text, images=raw_image, return_tensors="pt", padding=True)

    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    print(logits_per_image)
    probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
    print(probs)
    if probs[0][0] > probs[0][1]:
        print(text[0])
    else:
        print(text[1])


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
            # with torch.inference_mode():
            #     env0 = OriginalEnvWrapper(env)
            #     env0 = RslRlVecEnvWrapper(env0)
            params = parts[1].rstrip(")").split(", ")
            # executable_code.append('env0 = OriginalHighEnvWrapper(env)')
            # executable_code.append('env0 = RslRlVecEnvWrapper(env0)')
            executable_code.append(f'time_out, task_finished = push_to(env0, "{params[0]}", {params[1]}, policies, final_goal, time)')
        elif action == "walk_to":
            # with torch.inference_mode():
            #     env0 = OriginalEnvWrapper(env)
            #     env0 = RslRlVecEnvWrapper(env0)
            params = parts[1].rstrip(")")
            executable_code.append('env0 = OriginalHighEnvWrapper(env)')
            executable_code.append('env0 = RslRlVecEnvWrapper(env0)')
            executable_code.append(f'walk_to(env0, {params}, policies, final_goal)')
        elif action == "navigate_to":
            # with torch.inference_mode():
            #     env0 = OriginalEnvWrapper(env)
            #     env0 = RslRlVecEnvWrapper(env0)
            params = parts[1].rstrip(")")
            # executable_code.append('env0 = OriginalHighEnvWrapper(env)')
            # executable_code.append('env0 = RslRlVecEnvWrapper(env0)')
            executable_code.append(f'time_out, task_finished = navigate_to(env0, {params}, policies, final_goal, time)')
        elif action == "climb_to":
            # with torch.inference_mode():
            #     env0 = ClimbEnvWrapper(env)
            #     env0 = RslRlVecEnvWrapper(env0)
            params = parts[1].rstrip(")")
            # executable_code.append('env0 = OriginalLowEnvWrapper(env)')
            # executable_code.append('env0 = RslRlVecEnvWrapper(env0)')
            executable_code.append(f'time_out, task_finished = climb_to(env0, {params}, policies, final_goal. time)')

    # Return the generated code
    return "\n".join(executable_code)


def extract_plan(text):
    start_marker = '[begin of plan]'
    end_marker = '[end of plan]'
    start_index = text.find(start_marker)
    end_index = text.find(end_marker)
    
    if start_index == -1 or end_index == -1:
        return "Detailed Plan section not found."
    
    # Extract the content between the markers
    detailed_plan = text[start_index + len(start_marker):end_index].strip()
    
    # Split the detailed plan into individual plans
    # plans = detailed_plan.split("],")
    # plans = [plan.strip() + ']' for plan in plans if plan.strip()]
    
    return detailed_plan

def extract_detailed_plan_eval(text):
    start_marker = '[the optimal detailed plan]'
    end_marker = '[end of detailed plan]'
    start_index = text.find(start_marker)
    end_index = text.find(end_marker)
    
    if start_index == -1 or end_index == -1:
        return "Detailed Plan and evaluation section not found."
    
    # Extract the content between the markers
    plan_eval = text[start_index + len(start_marker):end_index].strip()
    
    return plan_eval

def extract_calculated_positions(text):
    start_marker = '[begin of summary]'
    end_marker = '[end of summary]'
    start_index = text.find(start_marker)
    end_index = text.find(end_marker)
    
    if start_index == -1 or end_index == -1:
        return "Calculated positions section not found."
    
    # Extract the content between the markers
    plan_eval = text[start_index + len(start_marker):end_index].strip()
    
    return plan_eval

class TreeNode:
    def __init__(self, state, layer=-1):
        self.state = state
        self.children = {}
        self.actions = set()
        self.difficulty = []
        self.num = 0
        self.r = 0 # immediate reward
        self.value = 0 # expected value
        self.layer = layer
    
def build_combined_tree(plans):
    root = TreeNode('root')  # Use a virtual root node
    
    for plan_steps in plans.values():
        current_node = root
        history = []
        
        for step_id, action, transition, difficulty_str in plan_steps:
            start, end = transition.split('->')
            difficulty = float(difficulty_str.split(': ')[1])
            history.append(action)
            history_tuple = tuple(history)
            
            if history_tuple not in current_node.children:
                current_node.children[history_tuple] = TreeNode(start, current_node.layer + 1)
            
            current_node = current_node.children[history_tuple]
            current_node.actions.add((step_id, action))
            current_node.difficulty.append(difficulty)
            current_node.num += 1
            cost = 1 - math.exp(math.log(2) * difficulty) -1 # -1 is step cost
            current_node.r = current_node.r + (cost - current_node.r) / current_node.num
    
    return list(root.children.values())[0] if root.children else None

def calculate_value_for_tree(node, terminal_reward=5, gamma=0.9):
    # If the node has no children, it is a terminal node
    if not node.children:
        node.value = node.r + terminal_reward
    else:
        # Calculate value for each child first
        for child in node.children.values():
            calculate_value_for_tree(child, terminal_reward)
        
        total_value = 0
        for child in node.children.values():
            total_value += child.value * gamma
        
        node.value = node.r + total_value / len(node.children)

def print_tree(node, depth=0):
    actions_str = ", ".join(f"{action}" for step_id, action in node.actions)
    print(f"Layer: {node.layer}, State: {node.state}, Actions: {actions_str}, Num: {node.num}, Cost: {node.r:.2f}, Value: {node.value:.2f}")
    for child in node.children.values():
        print_tree(child, depth + 1)

def print_tree2(node, depth=0):
    lines = []
    actions_str = ", ".join(f"{action}" for step_id, action in node.actions)
    lines.append(f"Layer: {node.layer}, State: {node.state}, Actions: {actions_str}, Cost: {node.r:.2f}")
    for child in node.children.values():
        lines.append(print_tree2(child, depth + 1))
    return "\n".join(lines)

def find_optimal_path(node):
    best_path = []

    current_node = node
    while current_node:
        best_path.append(current_node)
        if not current_node.children:
            break
        # Select the child with the highest value
        current_node = max(current_node.children.values(), key=lambda child: child.value)
    
    return best_path

def plot_tree(str):
    # 解析输入字符串
    pattern = re.compile(r"Layer: (\d+), State: (.*?), Actions: (.*?), Cost: ([\d.-]+)")
    matches = pattern.findall(str)

    # 创建有向图
    dot = Digraph()

    # 存储每层的节点列表
    nodes_in_layer = {}

    # 解析并添加节点和边
    for index, (layer, state, action, reward) in enumerate(matches):
        node_id = f"L{layer}_{action}_{index}"
        label = f"{state}, {action}\nCost: [{reward}]"
        dot.node(node_id, label)

        # 连接到上一个层级的节点
        layer_num = int(layer)
        if layer_num > 0:
            # 找到上一个层级的最后一个节点
            parent_node = nodes_in_layer.get(layer_num - 1, None)
            if parent_node:
                dot.edge(parent_node, node_id)

        # 更新当前层的最后一个节点
        nodes_in_layer[layer_num] = node_id


    # 显示图
    dot.render('tree', view=True, format='png')

def get_observation(env, objects_all):
    print("---------------------------------")
    current_object, objects_all = get_object_description(env, objects_all)
    print("Current visible objects:")
    print(current_object)

    object_description = "\n".join(objects_all.values())
    print("Objects description:")
    print(object_description)

    return current_object, objects_all

def LLM_task_planning(object_description, final_goal):
    prompt_proposer = f"""
    You are a quadrupedal robot on the ground in a 3D world. The ground has a height of 0.0m. Your x-y position is [0.0, 0.0]. There is a box_1, a box_2 and a box_3 on the ground. Object list = ['box_1', 'box_2', 'box_3']. Your goal is to navigate to a specific point in the 3D space. Your navigation goal is move to the top of platform.

    There are several objects in the scene that you may utilize. We use two parameters, position and size, to represent the location and size of an object, respectively. 
    Each object's position is represented by a 3D vector [x, y, z]. The axes are perpendicular to each other. The z axis is perpendicular to the ground surface and pointing upwards. Each object'size is represented as a 3D vector [length, width, height].
    
    {object_description}

    You have a skill set containing the following skills and corresponding parameters:

    * 'navigate_to_position(target_position)': Navigate to a target position. You can only move to positions on top of the same surface. For example, if you are currently on the ground, you cannot 'navigate' to the top of box_1, as the top of box_1 is a different surface.
    * 'climb_up_to_position(target_position)': Climb up to a target position. You can move to a position on top of a different surface where the z-value difference is less than 0.3. After you climb onto a new object, your surface changes to the object.
    * 'climb_down_to_position(target_position)': Climb down to a target position. The z-value difference is less than 0.3. After you climb down to ground, your surface changes to ground.
    * 'push_to_position(object_name, target_object_position)': Push a movable/unknown object only when you and the object are on top of the same surface. It handles walking to the object and pushing it to the target position.

    Give me five different abstract plans for using objects to help you complete navigation tasks. The plan must include which objects you need to use, the sequence you use the objects, how to use the objects. You must analyze the difficulty of the problem step by step and show the thinking process.

    You must follow these rules:

    * You cannot climb up or down a surface with a height difference larger than 0.25 meters from your current surface. For example, you are on a surface of height 0.0, and you want to climb up to a surface of height 0.4. The height difference between the surface you are on and the target surface is 0.4 - 0.0 = 0.4m.
    * You can only push objects for interaction. Do not perform other operations on objects.
    * You must first consider using a single object, then using multiple objects.

    You must follow the following answer template: 
    [Analyze]

    [begin of plan]
    Plan1: I need to use [objects]. First, ... Second ...
    Plan2: ...
    [end of plan]

    Answer example:
    Object description:

    * [box_1]: <center>: [0.0, -1.2, 0.0]; <size>: [0.8, 0.8, 0.18]. <movable>: True; <on_top_of>: ground; <adjacent_to>: None
    * [box_2]: <center>: [-1.0, -4.0, 0.0]; <size>: [0.8, 0.8, 0.26]. <movable>: True; <on_top_of>: ground; <adjacent_to>: None
    * [platform]: <center>: [-3.5, -1.5, 0.0]; <size>: [1.0, 1.0, 0.4]. <movable>: False; <on_top_of>: ground; <adjacent_to>: None

    Output:
    [Analyze]: The platform is immovable and has a height of 0.4 meters, which is too high to climb directly from the ground (0.0 meters) due to our 0.25-meter climbing limit. We need to use the movable boxes to create an intermediate step that allows us to climb onto the platform. 
    Object Analysis:
        box_1: Height is 0.18 meters. 0.18m - 0m = 0.18m and 0.4m - 0.18m = 0.22m, they are all less than 0.25m.
        box_2: Height is 0.26 meters. we cannot directly climb to box_2 since the height difference is 0.26m larger than 0.25m. 
    We can use Box 1 to create a path to the platform, or use Box 1 and Box 2 to create a stair to the platform. If I want to create a stair, I need to push taller box close to the platform, then push other boxes close to the tall box.

    [begin of plan]
    Plan1: I need to use [box_1]. Push box_1 close to the platform as a step to help climb to the platform.
    Plan2: I need to use [box_1, box_2]. First, push box_2 close to the platform. Second, push box_1 close to box_2 to create a stair.
    ...
    [end of plan]
    """

    start_proposer = time.time()
    llm_proposer, tokens_prompt_proposer, tokens_completion_proposer = chat_completions(prompt_proposer)
    print(llm_proposer)
    llm_plan = extract_plan(llm_proposer)
    end_proposer = time.time()
    time_proposer = end_proposer - start_proposer
    print(f"Time for proposer: {time_proposer}")

    # print(llm_plan)

    # for i, plan in enumerate(llm_plan):
    #     print(f"Plan {i+1}:")
    #     print(plan)
    plans = {}

    start_evaluator = time.time()

    prompt_evaluator = f"""
    You are a quadrupedal robot on the ground in a 3D world. The ground has a height of 0.0m. Your x-y position is [0.0, 0.0]. There is a box_1, a box_2 and a box_3 on the ground. Object list = ['box_1', 'box_2', 'box_3']. Your goal is to navigate to a specific point in the 3D space. Your navigation goal is move to the top of the platform.

    There are several objects in the scene that you may utilize. We use two parameters, position and size, to represent the location and size of an object, respectively. 
    Each object's position is represented by a 3D vector [x, y, z]. The axes are perpendicular to each other. The z axis is perpendicular to the ground surface and pointing upwards. Each object'size is represented as a 3D vector [length, width, height].
    
    {object_description}

    Here are some detailed plans to accomplish the navigation task. 
    {llm_plan}

    You need to evaluate these plans and choose the optimal plan that is feasible and has the lowest difficulty. You must evaluate the difficulty ranging from 0 to 1 for each step in every plan based on following criteria. 

    evaluation criteria:
    * First, whether the steps in the plan satisfy the given constraints and is feasible. The optimal plan must be feasible.
    * If a plan is feasible, you need to evaluate the difficulty of each step based on the type of action and its parameters. For example, for walking, the cost increases with walking distance. For climbing, the cost rises with height difference. For pushing, the cost increases as the distance the object is moved increases. Interactions involving objects typically have a higher difficulty.

    Constraints you must follow:

    * You cannot climb up to a surface that is higher than 0.3 meters above the surface you are currently on. For example, you are on a surface of height 0.05, and you want to climb up to a surface of height 0.4. The height difference between the surface you are on and the target surface is 0.4 - 0.05 = 0.35m. If the difference is higher than 0.3, the climb step is infeasible.
    * You can push an object only if it meets all the following two conditions simultaneously: first, You are on the surface of an immovable object or ground; second, you and the object are on top of the same surface, and you are not on top of the object it is pushing.
    * If you want to climb from one object to another object, you must make sure that 1) these two objects are adjacent. 2) the target surface must be 0.3 meters or less above the surface you are currently on. You must remember the <adjacent_to> information of each objects to judge whether two objects are adjacent. 

    After you choose the optimal plan, you need to generate detailed plans. Each step in the detailed plan consists of a skill with a abstract position like 'walk_to('abstract position')', the surface change like 'ground->ground' or 'ground->box_1'. The abstract position format uses prepositions and object words like 'close_to_[object]' and 'top_of_[object]'. You can only use 'close_to_' and 'top_of_' as prepositions.
    
    You have a skill library containing the following skills and corresponding parameters:

    * 'navigate_to(target_position)': Navigate to a target position. You can only move to positions on top of the same surface. For example, if you are currently on the ground, you cannot 'navigate' to the top of box_1, as the top of box_1 is a different surface.
    * 'climb_up_to(target_position)': Climb up to a target position. You can move to position on top of different surface where the z-value difference is less than 0.25. After you climb onto a new object, your surface changes to the object.
    * 'climb_down_to(target_position)': Climb down to a target position. After you climb down to a new surface, your surface changes to the object.
    * 'push_to(object_name, target_object_position)': Push a moveable/unknown object only when you and the object are on top of the same surface. It handles walking to the object and pushing it to the target position.

    You must follow the following answer template:
    [the optimal abstract plan]
    Plan...
    [end of abstract plan]

    [reason to choose the optimal plan]
    The optimal plan is Plan 1, which uses box_1 to create a path to the platform. The box_1 is the cloest to the platform and has the lowest difficulty.
 
    [the optimal detailed plan]
    Step0: start
    Step1: <skill>
    Step2: <skill>
    ...
    [end of detailed plan]

    Here is an example:
    [the optimal abstract plan]
    Plan: I need to use [box_1, box_2]. First, push box_2 near the platform. Next, push box_1 close to box_2. Climb onto box_1, then onto box_2, and finally onto the platform.
    [end of abstract plan]
 
    [the optimal detailed plan]
    Step0: start
    Step1: navigate_to(close_to_box_2)
    Step2: push_to(box_2, close_to_platform)
    Step3: navigate_to(close_to_box_1)
    Step4: push_to(box_1, close_to_box_2)
    Step5: climb_up_to(top_of_box_1)
    Step6: climb_up_to(top_of_box_2)
    Step7: climb_up_to(top_of_platform)
    [end of detailed plan]
    """
    llm_evaluator, tokens_prompt_evaluator, tokens_completion_evaluator = chat_completions(prompt_evaluator)
    print(llm_evaluator)

    end_evaluator = time.time()
    time_evaluator = end_evaluator - start_evaluator
    print(f"Time for evaluator: {time_evaluator}")

    llm_plan_eval = extract_detailed_plan_eval(llm_evaluator)
    # print(llm_plan_eval)

    # for i, plan in enumerate(llm_plan):

    #     prompt_evaluator = f"""
    #     You are a quadrupedal robot on the ground in a 3D world. The ground has a height of 0.0m. Your x-y position is [0.0, 0.0]. There is a box_1, a box_2 and a box_3 on the ground. Object list = ['box_1', 'box_2', 'box_3']. Your goal is to navigate to a specific point in the 3D space. Your navigation goal is move to the top of box_3.

    #     Numerical scene information:

    #     There are several objects in the scene that you may utilize. We use two parameters, position and size, to represent the location and size of an object, respectively. 
    #     Each object's position is represented by a 3D vector [x, y, z]. The axes are perpendicular to each other. The z axis is perpendicular to the ground surface and pointing upwards. Each object'size is represented as a 3D vector [length, width, height].
        
    #     {object_description}

    #     Here are some detailed plans to accomplish the navigation task. 
    #     {plan}

    #     You need to evaluate the difficulty ranging from 0 to 1 for each step in every plan based on following criteria. The closer the difficulty is to 1, the harder the step is to complete.

    #     difficulty criteria:
    #     * First, whether the step satisfies the given constraints and is feasible. If the step is not feasible, the difficulty is 1.
    #     * If a step is feasible, you need to determine its difficulty ranging from 0 to 1 based on the type of action and its parameters. For example, for walking, the cost increases with walking distance. For climbing, the cost rises with height difference. Interactions involving objects typically have a higher difficulty.

    #     Constraints you must follow:

    #     * You cannot climb up to a surface that is higher than 0.3 meters above the surface you are currently on. For example, you are on a surface of height 0.05, and you want to climb up to a surface of height 0.4. The height difference between the surface you are on and the target surface is 0.4 - 0.05 = 0.35m. If the difference is higher than 0.3, the climb step is infeasible.
    #     * You can push an object only if it meets all the following two conditions simultaneously: first, You are on the surface of an immovable object or ground; second, you and the object are on top of the same surface, and you are not on top of the object it is pushing.
    #     * If you want to climb from one object to another object, you must make sure that 1) these two objects are adjacent. 2) the target surface must be 0.3 meters or less above the surface you are currently on. You must remember the <adjacent_to> information of each objects to judge whether two objects are adjacent. 

    #     Important:
    #     * The push_to_position action creates new adjacency relationships. For example, push_to_position(box_1, position_beside_box_3) indicates that box_3 and box_1 are adjacent. You must pay attention to the result of each 'push_to_position' action to determine which objects have become adjacent. You can only climb between adjacent objects. You must remember which two objects are adjacent to avoid any contradictions.
    #     * You must understand which surface you are on to determine your height. For example, if you are on the ground, your height is 0. If you are on a object, your height is the height of the box. According to your height information to judge whether you can climb up to objects.
    #     * Note that the subsequent pushing strategy will alter the previous adjacency information.
    #     * At the start of each plan, none of the boxes are adjacent. The adjacency relationships are reinitialized, and you need to reassess them in each plan.

    #     You must follow the following answer template:
    #     [begin of evaluation]
    #     [
    #     ('step0', 'start', 'ground->ground', 'difficulty: ...'),
    #     ('step1', 'push_to_position(box_1, position_beside_box_3)', 'ground->ground', 'difficulty: ...'),
    #     ...
    #     ]
    #     [end of evaluation]

    #     Do not output anything other than the template. You need to fill in the difficulty for each step in the plan. 
    #     """
    #     llm_evaluator, tokens_prompt_evaluator, tokens_completion_evaluator = chat_completions(prompt_evaluator)
    #     # print(llm_evaluator)
    
    #     llm_plan_eval = extract_detailed_plan_eval(llm_evaluator)
    #     # print(llm_plan_eval)

    #     # print(ast.literal_eval(llm_plan_eval))  

    #     plans[f"Plan{i+1}"] = ast.literal_eval(llm_plan_eval)

    # end_evaluator = time.time()
    # time_evaluator = end_evaluator - start_evaluator
    # print(f"Time for evaluator: {time_evaluator}")

    # print(plans)

    # # Build the tree and calculate value
    # tree_root = build_combined_tree(plans)
    # calculate_value_for_tree(tree_root)

    # # Print the tree with calculated value
    # print_tree(tree_root)
    # print("\n")
    # tree_str = print_tree2(tree_root)

    # plot_tree(tree_str)

    # optimal_path = find_optimal_path(tree_root)

    # print("Optimal Path:")
    # for node in optimal_path:
    #     actions_str = ", ".join(f"{action}" for step_id, action in node.actions)
    #     print(f"Layer: {node.layer}, Actions: {actions_str}")


    # prompt_calculator = f"""
    # You are a quadrupedal robot on the ground in a 3D world. The ground has a height of 0.0m. Your x-y position is [0.0, 0.0]. There is a box_1, a box_2 and a box_3 on the ground. Object list = ['box_1', 'box_2', 'box_3']. Your goal is to navigate to a specific point in the 3D space. Your navigation goal is move to the top of box_3. 

    # Numerical scene information:

    # Each objects' center is represented by a 3D vector [x, y, z]. The axes are perpendicular to each other. z axis is perpendicular to the ground surface and pointing upwards.
    # Each object occupies a bounding box with size represented as <size>: [length, width, height].
    # {object_description}

    # You have a skill set containing the following skills and corresponding parameters:

    # * 'walk_to_position(target_position)': Walk to a target position. 
    # * 'climb_up_to_position(target_position)': Climb up to a target position. 
    # * 'climb_down_to_position(target_position)': Climb down to a target position. 
    # * 'push_to_position(object_name, target_object_position)': Push a moveable/unknown object to the target object position.

    # Your plan is: ('start', 'push_to_position(box_1, position_beside_box_3)', 'walk_to_position(position_beside_box_1)', 'climb_up_to_position(top_of_box_1)', 'climb_up_to_position(top_of_box_3)')

    # You help me to calculate the 3d position of all abstract target_position/target_object_position in the plan.

    # Common Rules:
    # * Calculate step by step and show the calculation process between <start of description> and <end of description>.
    # * Return the 3D position between <start of answer> and <end of answer>.
    # * You must not assume any position and directly query the updated position of the objects.
    # * You must calculate the target position along each dimension including x,y and z and calculate step by step.
    # * You must notice that after pushing, the position of object will change.

    # Responce example: 
    # <Step>: push_to_position(box_1, position_beside_box_3)
    # <start of description>
    # * Based on the position and size of objects, it is clear that there are no objects in the negative x direction of box_3, we can push box_1 to the negative x direction of the box_3. The target position along the x-axis is pos_box_3[0] - box_3_size[0]/2 - box_1_size[0]/2.
    # <end of description>
    # <start of answer>
    # The 3D target position of 'position_beside_box_3' is [pos_box_3[0] - box_3_size[0]/2 - box_1_size[0]/2, pos_box_3[1], pos_box_3[2]].
    # <end of answer>

    # <Step>: walk_to_position(position_beside_box_1)
    # <start of description>
    # ...

    # At last you give me a summarize like:
    # [begin of summary]:
    # position_beside_box_3: [x, y, z]
    # ...
    # [end of summary]:
    # """

    # start_calculator = time.time()
    # llm_calculator, tokens_prompt_calculator, tokens_completion_calculator = chat_completions(prompt_calculator)
    # end_calculator = time.time()
    # time_calculator = end_calculator - start_calculator
    # print("\n")
    # print(f"Time for calculator: {time_calculator}")

    # # print(llm_calculator)

    # calculated_positions = extract_calculated_positions(llm_calculator)

    # print(calculated_positions)

    # tokens_prompt_all = tokens_prompt_proposer + tokens_prompt_evaluator + tokens_prompt_calculator 
    # tokens_completion_all = tokens_completion_proposer + tokens_completion_evaluator + tokens_completion_calculator
    # cost_all = tokens_prompt_all * 0.0025 / 1000 + tokens_completion_all * 0.010 / 1000
    # print(f"Total cost: ${cost_all}")

    # calculated_positions_dict = {}

    # return optimal_path
    return llm_plan_eval