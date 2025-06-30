# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
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
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
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
import os
import torch

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
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp.observations import object_shape_pc, object_shape_pc_optimized

def visualize_point_cloud(
    env,
    object_name="interactive_object",
    num_points=1024,
    min_height=-0.5,
    max_height=0.5,
    num_envs_to_show=6,
    save_path="/tmp/point_cloud_test.png",
    show_plot=True
):
    """可视化物体点云的函数。
    
    Args:
        env: 环境实例
        object_name: 要可视化的物体名称
        num_points: 点云中的点数量
        min_height: 高度过滤最小值
        max_height: 高度过滤最大值
        num_envs_to_show: 要可视化的环境数量
        save_path: 图像保存路径
        show_plot: 是否显示图像
        
    Returns:
        point_cloud_3d: 形状为 (num_envs, num_points, 3) 的点云数据张量
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    
    # 获取原始环境实例
    unwrapped_env = getattr(env, "unwrapped", env)
    
    # 导入必要的函数
    from omni.isaac.lab.managers import SceneEntityCfg
    
    # 创建物体配置
    object_cfg = SceneEntityCfg(object_name)
    
    # 获取点云数据
    print(f"Getting point cloud for '{object_name}' with {num_points} points...")
    print(f"Height range: [{min_height}, {max_height}]")
    # object_shape_pc
    pc_flat = object_shape_pc(unwrapped_env, object_cfg, num_points, min_height, max_height)
    
    # 重塑为3D点云格式
    point_cloud_3d = pc_flat.reshape(-1, num_points, 3)
    
    # 限制环境数量
    num_envs_to_show = min(num_envs_to_show, unwrapped_env.num_envs)
    
    # 创建图形
    fig = plt.figure(figsize=(15, 5 * num_envs_to_show))
    
    for env_id in range(num_envs_to_show):
        # 获取点云
        points = point_cloud_3d[env_id].cpu().numpy()
        
        # 获取物体类型信息
        object_type = "Unknown"
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            prim_path = unwrapped_env.scene[object_name].root_physx_view.prim_paths[env_id]
            prim = stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                children = list(prim.GetChildren())
                mesh_children = [child for child in children if "Looks" not in str(child.GetPath())]
                if mesh_children:
                    object_type = mesh_children[0].GetName()
        except Exception as e:
            print(f"Error getting object type: {e}")
        
        # 创建三个视图
        ax1 = fig.add_subplot(num_envs_to_show, 3, env_id*3 + 1)
        ax2 = fig.add_subplot(num_envs_to_show, 3, env_id*3 + 2)
        ax3 = fig.add_subplot(num_envs_to_show, 3, env_id*3 + 3, projection='3d')
        
        # 设置标题
        fig.suptitle(f"Object Point Cloud Visualization (Height Range: {min_height}-{max_height}m)", fontsize=16)
        
        # 俯视图 (XY平面)
        sc1 = ax1.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap='viridis', s=40, alpha=0.8)
        ax1.set_title(f'Env {env_id} - {object_type} - Top View')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.grid(True)
        ax1.axis('equal')
        plt.colorbar(sc1, ax=ax1, label='Height (Z)')
        
        # 侧视图 (XZ平面)
        sc2 = ax2.scatter(points[:, 0], points[:, 2], c=points[:, 1], cmap='plasma', s=40, alpha=0.8)
        ax2.set_title(f'Env {env_id} - Side View')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.grid(True)
        ax2.axis('equal')
        plt.colorbar(sc2, ax=ax2, label='Y')
        
        # 3D视图
        sc3 = ax3.scatter(points[:, 0], points[:, 1], points[:, 2], 
                         c=points[:, 2], cmap='viridis', s=30, alpha=0.8)
        ax3.set_title('3D View')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.grid(True)
        
        # 添加坐标系参考线
        if len(points) > 0:
            # 计算点云包围盒
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            center = (min_coords + max_coords) / 2
            
            # 设置更好的视角
            max_range = np.max(max_coords - min_coords) * 0.6
            ax3.set_xlim(center[0] - max_range, center[0] + max_range)
            ax3.set_ylim(center[1] - max_range, center[1] + max_range)
            ax3.set_zlim(center[2] - max_range, center[2] + max_range)
            
            # 设置视角
            ax3.view_init(elev=30, azim=45)
        
        # 输出点云统计信息
        print(f"\nPoint Cloud Stats - Env {env_id} - {object_type}:")
        print(f"  Points count: {len(points)}")
        if len(points) > 0:
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            range_coords = max_coords - min_coords
            print(f"  X range: [{min_coords[0]:.3f}, {max_coords[0]:.3f}], width: {range_coords[0]:.3f}")
            print(f"  Y range: [{min_coords[1]:.3f}, {max_coords[1]:.3f}], length: {range_coords[1]:.3f}")
            print(f"  Z range: [{min_coords[2]:.3f}, {max_coords[2]:.3f}], height: {range_coords[2]:.3f}")
            
            # 计算点云统计数据
            mean = np.mean(points, axis=0)
            median = np.median(points, axis=0)
            
            # 按高度分布统计
            height_distribution = points[:, 2]
            height_quartiles = np.percentile(height_distribution, [25, 50, 75])
            
            print(f"  Mean position: [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]")
            print(f"  Height quartiles: 25%={height_quartiles[0]:.3f}, 50%={height_quartiles[1]:.3f}, 75%={height_quartiles[2]:.3f}")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # 保存图像
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPoint cloud image saved to: {save_path}")
    
    # 显示图像
    if show_plot:
        plt.show()
    
    return point_cloud_3d


def main():
    """Play with RSL-RL agent."""
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
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # print("robot_data: ",env.unwrapped.scene["robot"].data.joint_names)
    # print("robot_data: ",env.unwrapped.scene["robot"].data.joint_stiffness)
    # print("robot_data: ",env.unwrapped.scene["robot"].data.joint_damping)
    # print("robot_data: ",env.unwrapped.scene["robot"].data.default_joint_pos)

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

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0

    # ===== 添加点云测试代码 =====
    # print("Testing point cloud visualization...")
    # try:
        
    #     test_configs = [
    #         {"min_height": 0.0, "max_height": 0.5, "num_points": 64, "name": "few_points"},
    #         {"min_height": 0.0, "max_height": 0.5, "num_points": 1024, "name": "many_points"},
    #     ]
        
    #     for i, config in enumerate(test_configs):
    #         print(f"\nTesting configuration {i+1}/{len(test_configs)}: {config['name']}")
            
    #         # 调用可视化函数
    #         point_cloud = visualize_point_cloud(
    #             env=env.unwrapped,
    #             object_name="interactive_object",
    #             num_points=config["num_points"],
    #             min_height=config["min_height"],
    #             max_height=config["max_height"],
    #             num_envs_to_show=6,  # 仅显示前3个环境
    #             save_path=f"/tmp/point_cloud_{config['name']}.png",
    #             show_plot=True
    #         )
        
    #     # 等待用户确认继续
    #     input("\nPress Enter to continue the model playback process...")
        
    # except Exception as e:
    #     print(f"Point cloud visualization test error: {e}")
    #     import traceback
    #     traceback.print_exc()

    # ===== 点云测试代码结束 =====

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
            # obs[:, :3] = 0
            # obs[:, 48:235] = -0.08
            # print("actions: ", actions)
            # print("obs: ", obs)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
