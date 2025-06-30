# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to evaluate an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Evaluate an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during evaluation.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--evaluate", action="store_true", default=True, help="Run in evaluation mode.")
parser.add_argument("--eval_episodes", type=int, default=50, help="Number of episodes to evaluate.")
parser.add_argument("--success_threshold", type=float, default=0.25, 
                    help="Distance threshold (in meters) to consider the task successful.")
parser.add_argument("--visualize", action="store_true", default=True, help="Visualize evaluation results.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import os
import torch
import time
import json
import datetime
import traceback
from collections import defaultdict

from rsl_rl.runners import OnPolicyRunner

from omni.isaac.lab.envs import DirectMARLEnv, multi_agent_to_single_agent
from omni.isaac.lab.utils.dict import print_dict

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)


def main():
    """Evaluate RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "evaluation"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during evaluation.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
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

    # initialize evaluation metrics
    eval_metrics = {
        "success_rate": 0.0,
        "collisions": defaultdict(int),
        "episodes_completed": 0,
        "episode_rewards": [],
        "episode_lengths": [],
        "distance_to_target": [],
        "success_threshold": args_cli.success_threshold,
        "total_time": 0.0
    }
    
    # run evaluation
    print(f"\n{'='*50}")
    print(f"Starting evaluation for {args_cli.eval_episodes} episodes...")
    print(f"Success threshold: {args_cli.success_threshold} meters")
    print(f"{'='*50}\n")
    
    # record evaluation start time
    eval_start_time = time.time()
    
    try:
        for episode in range(args_cli.eval_episodes):
            # reset environment
            obs, _ = env.get_observations()
            done = False
            episode_reward = 0.0
            step_count = 0
            episode_collisions = defaultdict(int)
            
            # get initial target position
            try:
                from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.commands import generated_commands
                target_pose = generated_commands(env.unwrapped, command_name="pose_command")[0]  # first env's target
                target_pos = target_pose[:2]  # only care about XY plane position
            except Exception as e:
                print(f"Error getting target pose: {e}")
                target_pos = np.array([0.0, 0.0])  # fallback target
                
            print(f"Episode {episode+1}/{args_cli.eval_episodes}, Target: {target_pos}")
            
            # run single episode
            while not done:
                with torch.inference_mode():
                    # agent stepping
                    actions = policy(obs)
                    # env stepping
                    obs, rewards, dones, info = env.step(actions)
                
                # update cumulative reward
                episode_reward += rewards[0].item()  # first env's reward
                step_count += 1
                
                # check if reached target (distance < threshold)
                try:
                    robot_pos = env.unwrapped.scene["robot"].get_world_poses()[0][:2]  # first env's robot position
                    distance_to_target = torch.norm(torch.tensor(robot_pos) - torch.tensor(target_pos)).item()
                except Exception as e:
                    print(f"Error calculating distance: {e}")
                    distance_to_target = float('inf')
                
                # check collisions
                try:
                    if hasattr(env.unwrapped, "scene") and "contact_forces" in env.unwrapped.scene:
                        contact_data = env.unwrapped.scene["contact_forces"].get_contact_data()
                        
                        # check collisions by body part
                        body_prefixes = {
                            "head_upper": "Head_upper",
                            "head_lower": "Head_lower",
                            "hip": "_hip",
                            "calf": "_calf",
                            "thigh": "_thigh", 
                            "base": "base"
                        }
                        
                        for body_key, prefix in body_prefixes.items():
                            for contact in contact_data:
                                body_name = contact.get("body_name", "")
                                if prefix in body_name and contact.get("is_active", False):
                                    episode_collisions[body_key] += 1
                                    break  # count each body part collision only once per step
                except Exception as e:
                    print(f"Error checking collisions: {e}")
                
                # check if episode is done
                done = bool(dones[0].item())  # first env's done signal
                
                # check if reached target or max steps
                if distance_to_target < args_cli.success_threshold or step_count >= args_cli.video_length:
                    break
            
            # update evaluation metrics
            success = distance_to_target < args_cli.success_threshold
            eval_metrics["success_rate"] += float(success)
            eval_metrics["episodes_completed"] += 1
            eval_metrics["episode_rewards"].append(episode_reward)
            eval_metrics["episode_lengths"].append(step_count)
            eval_metrics["distance_to_target"].append(distance_to_target)
            
            # update collision counts
            for body_part, count in episode_collisions.items():
                eval_metrics["collisions"][body_part] += count
            
            # output single episode results
            print(f"  Result: {'✓ Success' if success else '✗ Failure'}, Final distance: {distance_to_target:.3f}m")
            print(f"  Collisions: {sum(episode_collisions.values())}, Steps: {step_count}, Reward: {episode_reward:.2f}")
            print(f"  Collision details: {dict(episode_collisions)}")
            print("-" * 40)
        
    except Exception as e:
        print(f"Evaluation error: {e}")
        traceback.print_exc()
    
    # calculate final metrics
    if eval_metrics["episodes_completed"] > 0:
        eval_metrics["success_rate"] = eval_metrics["success_rate"] / eval_metrics["episodes_completed"] * 100.0
    eval_metrics["total_time"] = time.time() - eval_start_time
    
    # output evaluation results
    print(f"\n{'='*50}")
    print(f"Evaluation Results ({eval_metrics['episodes_completed']} episodes, {eval_metrics['total_time']:.1f}s)")
    print(f"{'='*50}")
    print(f"Success Rate: {eval_metrics['success_rate']:.2f}%")
    
    if eval_metrics["distance_to_target"]:
        print(f"Average Distance to Target: {np.mean(eval_metrics['distance_to_target']):.3f}m")
    
    if eval_metrics["episode_lengths"]:
        print(f"Average Episode Length: {np.mean(eval_metrics['episode_lengths']):.1f} steps")
    
    if eval_metrics["episode_rewards"]:
        print(f"Average Episode Reward: {np.mean(eval_metrics['episode_rewards']):.2f}")
    
    print("\nCollision Statistics:")
    total_collisions = sum(eval_metrics["collisions"].values())
    
    for body_part, count in eval_metrics["collisions"].items():
        percentage = (count / total_collisions * 100) if total_collisions > 0 else 0
        print(f"  - {body_part.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    print(f"  - Total Collisions: {total_collisions}")
    
    if eval_metrics["episodes_completed"] > 0:
        print(f"  - Collisions per Episode: {total_collisions / eval_metrics['episodes_completed']:.2f}")
    
    # save evaluation results
    results_dir = os.path.join(log_dir, "eval_results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(results_dir, f"eval_results_{timestamp}.json")
    
    with open(result_file, "w") as f:
        json.dump({k: v if not isinstance(v, defaultdict) else dict(v) for k, v in eval_metrics.items()}, f, indent=2)
    
    print(f"\nEvaluation results saved to: {result_file}")
    
    # visualize evaluation results
    if args_cli.visualize:
        try:
            import matplotlib.pyplot as plt
            
            # create plots
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            
            # reward distribution
            if eval_metrics["episode_rewards"]:
                axs[0, 0].hist(eval_metrics["episode_rewards"], bins=10, alpha=0.7)
                axs[0, 0].set_title('Episode Rewards')
                axs[0, 0].set_xlabel('Reward')
                axs[0, 0].set_ylabel('Frequency')
            else:
                axs[0, 0].set_title('No Reward Data')
                axs[0, 0].axis('off')
            
            # step distribution
            if eval_metrics["episode_lengths"]:
                axs[0, 1].hist(eval_metrics["episode_lengths"], bins=10, alpha=0.7)
                axs[0, 1].set_title('Episode Lengths')
                axs[0, 1].set_xlabel('Steps')
                axs[0, 1].set_ylabel('Frequency')
            else:
                axs[0, 1].set_title('No Step Data')
                axs[0, 1].axis('off')
            
            # distance distribution
            if eval_metrics["distance_to_target"]:
                axs[1, 0].hist(eval_metrics["distance_to_target"], bins=10, alpha=0.7)
                axs[1, 0].set_title('Distance to Target')
                axs[1, 0].set_xlabel('Distance (m)')
                axs[1, 0].set_ylabel('Frequency')
                
                # Add vertical line at success threshold
                axs[1, 0].axvline(x=args_cli.success_threshold, color='r', linestyle='--', 
                                 label=f'Success threshold: {args_cli.success_threshold}m')
                axs[1, 0].legend()
            else:
                axs[1, 0].set_title('No Distance Data')
                axs[1, 0].axis('off')
            
            # collision pie chart
            collision_parts = list(eval_metrics["collisions"].keys())
            collision_counts = [eval_metrics["collisions"][part] for part in collision_parts]
            
            if sum(collision_counts) > 0:  # only draw pie chart if there are collisions
                axs[1, 1].pie(collision_counts, labels=[p.replace('_', ' ').title() for p in collision_parts], 
                             autopct='%1.1f%%', startangle=90)
                axs[1, 1].axis('equal')
                axs[1, 1].set_title('Collision Distribution')
            else:
                axs[1, 1].text(0.5, 0.5, "No collisions recorded", ha='center', va='center')
                axs[1, 1].axis('off')
            
            # add success rate as suptitle
            fig.suptitle(f'Evaluation Results: {eval_metrics["success_rate"]:.1f}% Success Rate', fontsize=16)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # save plot
            plt_file = os.path.join(results_dir, f"eval_visualization_{timestamp}.png")
            plt.savefig(plt_file)
            print(f"Visualization saved to: {plt_file}")
            
            # show plot
            plt.show()
            
        except Exception as e:
            print(f"Error creating visualization: {e}")
            traceback.print_exc()
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()