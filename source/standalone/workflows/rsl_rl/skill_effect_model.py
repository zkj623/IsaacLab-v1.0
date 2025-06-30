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
parser = argparse.ArgumentParser(description="Train and test skill effect prediction model.")
parser.add_argument("--mode", type=str, default="all", choices=["collect", "train", "test", "all"],
                    help="Mode of operation: collect data, train model, test model, or all")
parser.add_argument("--data_path", type=str, default="data/skill_effect/skill_effect_data_complete.pkl",
                    help="Path to dataset file")
parser.add_argument("--model_path", type=str, default=None,
                    help="Path to pre-trained model file")
parser.add_argument("--disable_fabric", action="store_true", default=False, 
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Unitree-Go2-v0", help="Name of the task.")

# 解析参数
args = parser.parse_args()

# 添加RSL-RL和AppLauncher参数
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动Omniverse应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import gymnasium as gym
import time
import json
import datetime
import pickle
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from scipy.ndimage import binary_dilation
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from collections import defaultdict

from omni.isaac.lab.app import AppLauncher
import omni.isaac.lab_tasks
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab_tasks.utils import get_checkpoint_path, parse_env_cfg
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import cli_args

class SkillEffectDataset(Dataset):
    """数据集类，用于存储技能执行结果数据"""
    
    def __init__(self, features, success_labels, cost_labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.success_labels = torch.tensor(success_labels, dtype=torch.float32)
        self.cost_labels = torch.tensor(cost_labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.success_labels[idx], self.cost_labels[idx]

class SkillEffectModel(nn.Module):
    """预测技能的成功率和代价的神经网络模型"""
    
    def __init__(self, input_dim=325, hidden_dim=256):
        super(SkillEffectModel, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 成功率预测（分类任务）
        self.success_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # 代价预测（回归任务）
        self.cost_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        shared_features = self.shared_layers(x)
        success_prob = self.success_head(shared_features)
        cost = self.cost_head(shared_features)
        return success_prob, cost

class PointCloudSkillEffectModel(nn.Module):
    """处理点云数据的技能效果模型"""
    
    def __init__(self, lidar_points=676, obs_dim=0):
        super(PointCloudSkillEffectModel, self).__init__()
        
        # 处理目标位置
        self.target_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )
        
        # 处理点云数据的PointNet风格网络
        self.point_encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),  # 输入通道是3(xyz)
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.ReLU()
        )
        
        # 点云的MaxPool聚合
        self.pool = lambda x: torch.max(x, 2)[0]
        
        # 处理基本观测(如果有)
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        ) if obs_dim > 0 else None
        
        # 特征融合
        fusion_dim = 64 + 256 + (64 if obs_dim > 0 else 0)
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # 输出头
        self.success_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出范围0-1
        )
        
        self.cost_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 将代价也限制在0-1范围
        )
    
    def forward(self, data_dict):
        """前向传播，接受包含多个特征源的字典"""
        # 处理目标位置和点云数据
        if isinstance(data_dict, dict):
            # 字典输入模式
            target = data_dict["target"]
            points = data_dict["lidar"]
            base_obs = data_dict.get("base_observation", None)
        else:
            # 兼容模式: 假设输入是一个包含所有特征的平坦向量
            # 前2个元素是目标位置，后面是点云数据
            target = data_dict[:, :2]
            points = data_dict[:, 2:].view(data_dict.shape[0], -1, 3)
            base_obs = None
        
        batch_size = target.shape[0]
        
        # 编码目标位置 [B, 2] -> [B, 64]
        target_features = self.target_encoder(target)
        
        # 编码点云 [B, N, 3] -> [B, 256]
        points = points.permute(0, 2, 1)  # [B, 3, N]
        point_features = self.point_encoder(points)   # [B, 256, N]
        point_features = self.pool(point_features)    # [B, 256]
        
        # 融合特征
        features = [target_features, point_features]
        
        # 处理基本观测(如果有)
        if base_obs is not None and self.obs_encoder is not None:
            obs_features = self.obs_encoder(base_obs)
            features.append(obs_features)
        
        # 融合所有特征
        fused_features = torch.cat(features, dim=1)
        shared_features = self.fusion_net(fused_features)
        
        # 预测输出
        success_prob = self.success_head(shared_features)
        cost = self.cost_head(shared_features)
        
        return success_prob, cost

class SkillEffectModelTrainer:
    """技能效果模型训练器"""
    
    def __init__(self, model_save_dir="models/skill_effect"):
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)
        self.model = None
        self.scaler = None
        
    def train_model(self, dataset_path, batch_size=64, epochs=100, lr=0.001, test_size=0.2, device='cuda'):
        """训练模型"""
        print(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        features = data['features']
        success_labels = data['success_labels']
        cost_labels = data['cost_labels']
        
        print(f"Dataset size: {len(features)}")
        print(f"Features shape: {features.shape}")
        print(f"Success rate distribution: {np.mean(success_labels):.4f}")
        print(f"Cost mean: {np.mean(cost_labels):.4f}, std: {np.std(cost_labels):.4f}")
        
        # 拆分训练集和测试集
        X_train, X_test, y_success_train, y_success_test, y_cost_train, y_cost_test = train_test_split(
            features, success_labels, cost_labels, test_size=test_size, random_state=42
        )
        
        # 创建数据加载器
        train_dataset = SkillEffectDataset(X_train, y_success_train, y_cost_train)
        test_dataset = SkillEffectDataset(X_test, y_success_test, y_cost_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # 创建模型
        input_dim = features.shape[1]

        # model = SkillEffectModel(input_dim=input_dim)

        n_points = (input_dim - 2) // 3  # 准确计算点的数量
        model = PointCloudSkillEffectModel(lidar_points=n_points, obs_dim=0)
        model.to(device)
        
        # 定义优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=lr)
        success_criterion = nn.MSELoss()
        cost_criterion = nn.MSELoss()
        
        # 训练循环
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            train_success_loss = 0
            train_cost_loss = 0
            
            for batch_features, batch_success, batch_cost in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                batch_features = batch_features.to(device)
                batch_success = batch_success.to(device).unsqueeze(1)
                batch_cost = batch_cost.to(device).unsqueeze(1)
                
                # 梯度清零
                optimizer.zero_grad()
                
                # 前向传播
                batch_target = batch_features[:, :2]
                batch_lidar = batch_features[:, 2:].reshape(batch_features.shape[0], -1, 3)
                data_dict = {
                    "target": batch_target,
                    "lidar": batch_lidar
                }
                success_pred, cost_pred = model(data_dict)

                # success_pred, cost_pred = model(batch_features)
                
                # 计算损失
                success_loss = success_criterion(success_pred, batch_success)
                cost_loss = cost_criterion(cost_pred, batch_cost)
                
                # 总损失
                loss = success_loss + cost_loss
                
                # 反向传播
                loss.backward()
                
                # 更新权重
                optimizer.step()
                
                train_success_loss += success_loss.item()
                train_cost_loss += cost_loss.item()
            
            # 验证
            model.eval()
            val_success_loss = 0
            val_cost_loss = 0
            success_preds = []
            success_true = []
            cost_preds = []
            cost_true = []
            
            with torch.no_grad():
                for batch_features, batch_success, batch_cost in test_loader:
                    batch_features = batch_features.to(device)
                    batch_success = batch_success.to(device).unsqueeze(1)
                    batch_cost = batch_cost.to(device).unsqueeze(1)
                    
                    success_pred, cost_pred = model(batch_features)
                    
                    val_success_loss += success_criterion(success_pred, batch_success).item()
                    val_cost_loss += cost_criterion(cost_pred, batch_cost).item()
                    
                    # 收集预测和真实值
                    success_preds.extend(success_pred.cpu().numpy())
                    success_true.extend(batch_success.cpu().numpy())
                    cost_preds.extend(cost_pred.cpu().numpy())
                    cost_true.extend(batch_cost.cpu().numpy())
            
            # 计算验证指标 - 全部使用回归指标
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            success_mse = mean_squared_error(success_true, success_preds)
            success_mae = mean_absolute_error(success_true, success_preds) 
            success_r2 = r2_score(success_true, success_preds)
            
            cost_mse = mean_squared_error(cost_true, cost_preds)
            cost_mae = mean_absolute_error(cost_true, cost_preds)
            cost_r2 = r2_score(cost_true, cost_preds)
            
            # 输出训练信息
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train: success_loss: {train_success_loss/len(train_loader):.4f}, cost_loss: {train_cost_loss/len(train_loader):.4f}")
            print(f"  Val: success_loss: {val_success_loss/len(test_loader):.4f}, cost_loss: {val_cost_loss/len(test_loader):.4f}")
            print(f"  Success MSE: {success_mse:.4f}, MAE: {success_mae:.4f}, R²: {success_r2:.4f}")
            print(f"  Cost MSE: {cost_mse:.4f}, MAE: {cost_mae:.4f}, R²: {cost_r2:.4f}")
    
            # 早停
            val_loss = val_success_loss/len(test_loader) + val_cost_loss/len(test_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # 保存最佳模型
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                model_path = os.path.join(self.model_save_dir, f"skill_effect_model_{timestamp}.pt")
                torch.save(model.state_dict(), model_path)
                print(f"Model saved to {model_path}")
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # 加载最佳模型
        model_files = [f for f in os.listdir(self.model_save_dir) if f.endswith('.pt')]
        if model_files:
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join(self.model_save_dir, latest_model)
            model.load_state_dict(torch.load(model_path))
            print(f"Loaded best model from {model_path}")
        
        self.model = model
        return model

    def save_model(self, filename=None):
        """保存模型"""
        if self.model is None:
            print("No model to save")
            return
            
        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"skill_effect_model_{timestamp}.pt"
            
        model_path = os.path.join(self.model_save_dir, filename)
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
    def load_model(self, model_path, input_dim=191, device='cuda'):
        """加载已训练的模型"""
        # model = SkillEffectModel(input_dim=input_dim)
        model = PointCloudSkillEffectModel(lidar_points=676, obs_dim=0)
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        model.eval()
        self.model = model
        print(f"Model loaded from {model_path}")
        return model
    
    def predict(self, features):
        """预测技能成功率和代价"""
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        with torch.no_grad():
            # features_tensor = torch.tensor(features, dtype=torch.float32).to(next(self.model.parameters()).device)
            # success_prob, cost = self.model(features_tensor)

            if isinstance(features, np.ndarray):
                features_tensor = torch.tensor(features, dtype=torch.float32)
            else:
                features_tensor = features
                
            features_tensor = features_tensor.to(next(self.model.parameters()).device)
            
            # 为点云模型准备输入
            batch_target = features_tensor[:, :2]
            batch_lidar = features_tensor[:, 2:].reshape(features_tensor.shape[0], -1, 3)
            data_dict = {
                "target": batch_target,
                "lidar": batch_lidar
            }
            success_prob, cost = self.model(data_dict)
        
        return success_prob.cpu().numpy(), cost.cpu().numpy()

class SkillEffectDataCollector:
    """收集技能效果数据"""
    
    def __init__(self, env, policies, data_save_dir="data/skill_effect"):
        self.env = env
        self.policies = policies
        self.data_save_dir = data_save_dir
        os.makedirs(data_save_dir, exist_ok=True)
        
        # 初始化数据存储结构
        self.features = []  # 环境特征
        self.success_labels = []  # 成功标签
        self.cost_labels = []  # 代价标签
        self.skill_types = []  # 技能类型
        self.targets = []  # 目标位置
        
    def collect_data_for_skill(self, skill_name, num_trials=500, max_steps=100):
        """为特定技能收集数据"""
        print(f"Collecting data for skill: {skill_name}")
        policy = self.policies[skill_name]
        
        from omni.isaac.lab.managers import SceneEntityCfg
        from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import OriginalHighEnvWrapper, OriginalLowEnvWrapper
        
        env0 = self.env
        
        for trial in tqdm(range(num_trials), desc=f"Collecting {skill_name} data"):
            # 重置环境
            with torch.inference_mode():
                obs, _ = env0.reset()
            
            # 随机生成目标位置
            # 根据技能类型调整目标生成范围
            if skill_name == "push":
                target_x = np.random.uniform(-3, 3)
                target_y = np.random.uniform(-3, 3) 
                target_position = torch.tensor([target_x, target_y], device=self.env.unwrapped.device)
                object_name = "box_1"  # 假设要推动的物体名称
            elif skill_name == "climb":
                target_x = np.random.uniform(-3, 3)
                target_y = np.random.uniform(-3, 3)
                target_z = np.random.uniform(0.2, 0.6)
                target_position = torch.tensor([target_x, target_y, target_z], device=self.env.unwrapped.device)
            else:  # walk or navigate
                target_x = np.random.uniform(-3, 3)
                target_y = np.random.uniform(-3, 3)
                target_position = torch.tensor([target_x, target_y], device=self.env.unwrapped.device)
            
            # 获取机器人周围环境特征
            env_features = self._extract_features(obs, target_position)
            
            # 尝试执行技能
            success, steps_taken = self._execute_skill(env0, policy, obs, target_position, skill_name, max_steps)
            
            # 计算代价（时间步数）
            cost = steps_taken / max_steps  # 归一化的代价
            
            # 存储数据
            self.features.append(env_features)
            self.success_labels.append(float(success))
            self.cost_labels.append(cost)
            self.skill_types.append(skill_name)
            self.targets.append(target_position.cpu().numpy())
            
            # 每100次试验保存一次数据
            if (trial + 1) % 100 == 0:
                self.save_data(f"{skill_name}_partial_{trial+1}")
        
        # 最终保存数据
        self.save_data(f"{skill_name}_complete_{num_trials}")
        
    # def _extract_features(self, obs, target_position):
    #     """
    #     从观测中提取特征
    #     这里我们使用原始观测作为特征，包括：
    #     - 机器人姿态
    #     - 高度扫描数据
    #     - 位置信息
    #     """
    #     lidar_data = self.env.unwrapped.scene.sensors["height_scanner_wide"].data.ray_hits_w.cpu().numpy()
    #     lidar_data = lidar_data.flatten()
    #     print("Lidar data shape:", lidar_data.shape)
    #     print("Lidar data:", lidar_data)

    #     # 这里假设obs是一个tensor，提取所有相关的观测特征
    #     features = obs[0, 6:].cpu().numpy()  # 取第一个环境的观测
    #     # features[2:] = lidar_data.flatten()  # 高度扫描数据

    #     # 获取机器人位置
    #     robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
        
    #     # 基础观察特征
    #     height_features = lidar_data
        
    #     # 计算目标相对位置 (作为显式特征)
    #     target_rel_pos = target_position.cpu().numpy()[:2] - robot_pos[:2]
        
    #     # 将相对位置显式添加为特征
    #     combined_features = np.concatenate([
    #         target_rel_pos,  # 显式目标相对位置 [x, y]
    #         height_features    # 原始观测特征
    #     ])
        
    #     return features

    def _extract_features(self, obs, target_position):
        """
        从观测中提取特征
        这里我们保留点云的3D结构
        """
        # 获取LiDAR数据
        lidar_data = self.env.unwrapped.scene.sensors["height_scanner_wide"].data.ray_hits_w.cpu().numpy()
        
        # 获取机器人位置
        robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
        
        # 计算目标相对位置 (作为显式特征)
        target_rel_pos = target_position.cpu().numpy()[:2] - robot_pos[:2]
        
        # 为了保持与现有代码的兼容性，将所有特征合并为一个向量
        # 前2个元素是目标相对位置，后面是展平的LiDAR数据
        combined_features = np.concatenate([
            target_rel_pos.flatten(),  # 目标相对位置 [2]
            lidar_data.flatten()       # 展平的LiDAR数据 [N*3]
        ])
        
        return combined_features
    
    def _execute_skill(self, env0, policy, obs, target_position, skill_name, max_steps):
        """
        执行技能并返回是否成功以及花费的步数
        """
        steps = 0
        task_finished = False
        success_rate = 0
        
        with torch.inference_mode():
            while steps < max_steps and not task_finished:
                # 根据技能类型处理观测
                # if skill_name == "push":
                #     # 准备推动技能的观测
                #     obs_push = self._prepare_push_obs(obs, target_position)
                #     actions = policy(obs_push)
                # elif skill_name == "climb":
                #     # 准备攀爬技能的观测
                #     obs_climb = self._prepare_climb_obs(obs, target_position)
                #     actions = policy(obs_climb)
                # elif skill_name == "walk":
                #     # 准备行走技能的观测
                #     obs_walk = self._prepare_walk_obs(obs, target_position)
                #     actions = policy(obs_walk)
                # else:  # navigate
                #     # 准备导航技能的观测
                #     obs_nav = self._prepare_nav_obs(obs, target_position)
                #     actions = policy(obs_nav)
                
                # 执行动作
                actions = policy(obs)
                obs, _, _, _ = env0.step(actions)
                steps += 1
                
                # 检查是否完成任务
                # robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w
                
                # distance = torch.norm(robot_pos[0, :2] - target_position[:2])
                # task_finished = distance < 0.25

                dist_to_goal = torch.norm(obs[0, 6:8])
                task_finished = dist_to_goal < 0.5

                print("Distance to goal:", dist_to_goal.item())
                
        if task_finished:
            print(f"Task completed successfully in {steps} steps.")
            success_rate = 1.0
        else:
            print(f"Task failed.")
            if dist_to_goal > 2.0:
                success_rate = 0.0
            else:
                success_rate = np.exp(-2 * (dist_to_goal.cpu().numpy() - 0.5))
        
        return success_rate, steps
    
    def _prepare_walk_obs(self, obs, target_position):
        """准备行走技能的观测"""
        robot_state = self.env.unwrapped.scene["robot"].data.root_state_w
        
        # 计算机器人到目标的相对位置
        from omni.isaac.lab.utils.math import subtract_frame_transforms
        goal, _ = subtract_frame_transforms(robot_state[:, :3], robot_state[:, 3:7], target_position)
        
        # 对观测进行必要的处理
        obs_walk = torch.cat((obs[:, :3], obs[:, 10:17]), dim=1)
        
        # 设置目标位置
        obs_walk[:, 6:8] = goal[:, :2]
        obs_walk[:, 9] = 1.5  # 假设速度目标
        
        return obs_walk
    
    def _prepare_nav_obs(self, obs, target_position):
        """准备导航技能的观测"""
        robot_state = self.env.unwrapped.scene["robot"].data.root_state_w
        
        # 计算机器人到目标的相对位置
        from omni.isaac.lab.utils.math import subtract_frame_transforms
        goal, _ = subtract_frame_transforms(robot_state[:, :3], robot_state[:, 3:7], target_position)
        
        # 对观测进行必要的处理
        # obs_nav = torch.cat((obs[:, :3], obs[:, 10:17], obs[:, 59:246]), dim=1)
        
        # 设置目标位置
        # obs[:, 6:8] = goal[:, :2]
        
        return obs
    
    def save_data(self, suffix=""):
        """保存收集的数据"""
        dataset = {
            'features': np.array(self.features),
            'success_labels': np.array(self.success_labels),
            'cost_labels': np.array(self.cost_labels),
            'skill_types': self.skill_types,
            'targets': self.targets
        }
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"skill_effect_data_{suffix}.pkl"
        file_path = os.path.join(self.data_save_dir, filename)
        
        with open(file_path, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Data saved to {file_path}")
        print(f"Total samples: {len(self.features)}")
        print(f"Success rate: {np.mean(self.success_labels):.4f}")
        
    def load_data(self, file_path):
        """加载保存的数据"""
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f)
        
        self.features = list(dataset['features'])
        self.success_labels = list(dataset['success_labels'])
        self.cost_labels = list(dataset['cost_labels'])
        self.skill_types = list(dataset['skill_types'])
        self.targets = list(dataset['targets'])
        
        print(f"Data loaded from {file_path}")
        print(f"Total samples: {len(self.features)}")
        print(f"Success rate: {np.mean(self.success_labels):.4f}")

class SkillEffectVisualizer:
    """技能效果模型的可视化工具"""
    
    def __init__(self, env, model, device='cuda'):
        self.env = env
        self.model = model
        self.device = device
        
    def visualize_robot_surroundings(self, grid_size=0.1, range_radius=5.0,
                                    skill_name="navigate", show_plot=True, save_path=None):
        """可视化机器人周围区域的技能效果预测"""
        # 获取当前机器人观测
        obs, _ = self.env.get_observations()
        lidar_data = self.env.unwrapped.scene.sensors["height_scanner_wide"].data.ray_hits_w.cpu().numpy()
        
        # 获取机器人位置
        robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
        robot_rot = self.env.unwrapped.scene["robot"].data.root_quat_w[0].cpu().numpy()
        
        # 创建以机器人为中心的网格（而不是固定的绝对坐标）
        range_x = (robot_pos[0] - range_radius, robot_pos[0] + range_radius)
        range_y = (robot_pos[1] - range_radius, robot_pos[1] + range_radius)
        
        x = np.arange(range_x[0], range_x[1], grid_size)
        y = np.arange(range_y[0], range_y[1], grid_size)
        X, Y = np.meshgrid(x, y)
        
        grid_points = np.column_stack((X.flatten(), Y.flatten()))
        success_probs = np.zeros(len(grid_points))
        costs = np.zeros(len(grid_points))
        
        # 批处理预测
        batch_size = 256
        for i in range(0, len(grid_points), batch_size):
            batch_points = grid_points[i:i+batch_size]
            batch_features = []
            
            # 为每个网格点准备特征
            for point in batch_points:
                                # 更新特征中的目标位置为相对位置
                target_rel_pos = point - robot_pos[:2]  # 相对位置

                feature = np.concatenate([
                    target_rel_pos.flatten(),  # 目标相对位置 [2]
                    lidar_data.flatten()       # 展平的LiDAR数据 [N*3]
                ])

                # feature = obs[0, 6:].cpu().numpy()  # 取第一个环境的观测
                # feature[:2] = target_rel_pos.flatten()  # 更新目标相对位置

                # print("Feature shape:", feature.shape)
                
                batch_features.append(feature)
            
            # 进行预测
            batch_features = np.array(batch_features)
            batch_success_probs, batch_costs = self.model.predict(batch_features)
            
            # 更新结果
            success_probs[i:i+len(batch_features)] = batch_success_probs.flatten()
            costs[i:i+len(batch_features)] = batch_costs.flatten()
        
        # 重塑结果为网格形状
        success_probs_grid = success_probs.reshape(X.shape)
        costs_grid = costs.reshape(X.shape)
        
        # 可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # 成功率热图
        success_cmap = LinearSegmentedColormap.from_list('success_cmap', ['red', 'yellow', 'green'])
        success_img = ax1.imshow(success_probs_grid, extent=(range_x[0], range_x[1], range_y[0], range_y[1]), 
                                origin='lower', cmap=success_cmap, vmin=0, vmax=1)
        ax1.set_title(f'{skill_name.capitalize()} Skill - Success Rate')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        fig.colorbar(success_img, ax=ax1, label='Success Probability')
        
        # 标记机器人位置
        ax1.plot(robot_pos[0], robot_pos[1], 'bo', markersize=10, label='Robot')
        ax1.legend()
        
        # 绘制朝向箭头
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat([robot_rot[1], robot_rot[2], robot_rot[3], robot_rot[0]])  # wxyz to xyzw
        heading = r.as_euler('xyz')[2]
        dx = np.cos(heading) * 0.5
        dy = np.sin(heading) * 0.5
        ax1.arrow(robot_pos[0], robot_pos[1], dx, dy, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
        
        # 代价热图
        cost_cmap = LinearSegmentedColormap.from_list('cost_cmap', ['green', 'yellow', 'red'])
        cost_img = ax2.imshow(costs_grid, extent=(range_x[0], range_x[1], range_y[0], range_y[1]), 
                            origin='lower', cmap=cost_cmap, vmin=0, vmax=1)
        ax2.set_title(f'{skill_name.capitalize()} Skill - Execution Cost')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        fig.colorbar(cost_img, ax=ax2, label='Normalized Cost')
        
        # 标记机器人位置
        ax2.plot(robot_pos[0], robot_pos[1], 'bo', markersize=10, label='Robot')
        ax2.legend()
        ax2.arrow(robot_pos[0], robot_pos[1], dx, dy, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
        
        # 添加障碍物标记
        self._add_obstacles_to_plot(ax1)
        self._add_obstacles_to_plot(ax2)
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        # 显示图像
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        return success_probs_grid, costs_grid
    
    def _add_obstacles_to_plot(self, ax):
        """在图表上添加障碍物标记"""
        # 获取环境中的物体
        try:
            objects = self.env.unwrapped.scene.rigid_objects
            
            for obj_name, obj in objects.items():
                if obj_name != "robot":
                    # 获取物体位置和大小
                    object_pos = obj.data.root_pos_w[0].cpu().numpy()
                    
                    # 尝试获取物体尺寸
                    try:
                        object_size = obj.cfg.spawn.size
                        if not isinstance(object_size, tuple):
                            object_size = (0.5, 0.5)
                        
                        # 绘制矩形表示障碍物
                        from matplotlib.patches import Rectangle
                        rect = Rectangle(
                            (object_pos[0] - object_size[0]/2, object_pos[1] - object_size[1]/2),
                            object_size[0], object_size[1],
                            linewidth=1, edgecolor='black', facecolor='gray', alpha=0.6
                        )
                        ax.add_patch(rect)
                        
                        # 添加标签
                        ax.text(object_pos[0], object_pos[1], obj_name, 
                                ha='center', va='center', color='white', fontsize=8)
                    except:
                        # 如果无法获取尺寸信息，使用点标记
                        ax.plot(object_pos[0], object_pos[1], 'ks', markersize=6)
                        ax.text(object_pos[0], object_pos[1], obj_name, 
                                ha='center', va='bottom', color='black', fontsize=8)
        except Exception as e:
            print(f"Error adding obstacles to plot: {e}")
    
    def visualize_skill_comparison(self, target_position, grid_size=0.1, range_x=(-5, 5), range_y=(-5, 5),
                                  skills=["walk", "navigate", "push", "climb"], 
                                  show_plot=True, save_path=None):
        """比较不同技能对于同一目标的效果"""
        # 获取机器人位置
        robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
        
        # 创建子图
        fig, axes = plt.subplots(len(skills), 2, figsize=(14, 6*len(skills)))
        
        # 如果只有一个技能，确保axes是二维数组
        if len(skills) == 1:
            axes = np.array([axes])
        
        for i, skill in enumerate(skills):
            # 可视化这个技能
            success_grid, cost_grid = self.visualize_robot_surroundings(
                grid_size=grid_size, range_x=range_x, range_y=range_y,
                skill_name=skill, show_plot=False
            )
            
            # 获取这个子图的轴
            ax1, ax2 = axes[i]
            
            # 成功率热图
            success_cmap = LinearSegmentedColormap.from_list('success_cmap', ['red', 'yellow', 'green'])
            success_img = ax1.imshow(success_grid, extent=(range_x[0], range_x[1], range_y[0], range_y[1]), 
                                    origin='lower', cmap=success_cmap, vmin=0, vmax=1)
            ax1.set_title(f'{skill.capitalize()} Skill - Success Rate')
            ax1.set_xlabel('X (m)')
            ax1.set_ylabel('Y (m)')
            fig.colorbar(success_img, ax=ax1, label='Success Probability')
            
            # 代价热图
            cost_cmap = LinearSegmentedColormap.from_list('cost_cmap', ['green', 'yellow', 'red'])
            cost_img = ax2.imshow(cost_grid, extent=(range_x[0], range_x[1], range_y[0], range_y[1]), 
                                 origin='lower', cmap=cost_cmap, vmin=0, vmax=1)
            ax2.set_title(f'{skill.capitalize()} Skill - Execution Cost')
            ax2.set_xlabel('X (m)')
            ax2.set_ylabel('Y (m)')
            fig.colorbar(cost_img, ax=ax2, label='Normalized Cost')
            
            # 标记机器人位置和目标位置
            ax1.plot(robot_pos[0], robot_pos[1], 'bo', markersize=8, label='Robot')
            ax2.plot(robot_pos[0], robot_pos[1], 'bo', markersize=8, label='Robot')
            
            ax1.plot(target_position[0], target_position[1], 'ro', markersize=8, label='Target')
            ax2.plot(target_position[0], target_position[1], 'ro', markersize=8, label='Target')
            
            ax1.legend()
            ax2.legend()
            
            # 添加障碍物标记
            self._add_obstacles_to_plot(ax1)
            self._add_obstacles_to_plot(ax2)
        
        plt.tight_layout()
        
        # 保存图像
        # if save_path:
        #     plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #     print(f"Comparison visualization saved to {save_path}")
        
        # 显示图像
        if show_plot:
            plt.show()
        else:
            plt.close()

        
    def create_animation(self, skill_name="navigate", frames=50, interval=10, range_radius=5.0,
                         grid_size=0.1, save_path=None):
        """创建动态效果模型可视化"""
        import matplotlib.animation as animation
        
        obs, _ = self.env.get_observations()
        policy = self.env.policies[skill_name]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # 成功率和代价的颜色映射
        success_cmap = LinearSegmentedColormap.from_list('success_cmap', ['red', 'yellow', 'green'])
        cost_cmap = LinearSegmentedColormap.from_list('cost_cmap', ['green', 'yellow', 'red'])
        
        # 获取机器人位置
        robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
        range_x = (robot_pos[0] - range_radius, robot_pos[0] + range_radius)
        range_y = (robot_pos[1] - range_radius, robot_pos[1] + range_radius)

        # 初始化图像 - 先获取一次效果预测以显示初始状态
        success_grid, cost_grid = self.visualize_robot_surroundings(
            grid_size=grid_size, range_radius=range_radius,
            skill_name=skill_name, show_plot=False
        )

        # 初始化图像
        grid_shape = (int((2*range_radius)/grid_size), int((2*range_radius)/grid_size))
        white_success_grid = np.ones(grid_shape) * np.nan  # 使用NaN来创建透明背景
        white_cost_grid = np.ones(grid_shape) * np.nan     # 使用NaN来创建透明背景

        # 使用全白数组初始化图像
        success_img = ax1.imshow(white_success_grid, 
                            extent=(range_x[0], range_x[1], range_y[0], range_y[1]),
                            origin='lower', cmap=success_cmap, vmin=0, vmax=1)
        cost_img = ax2.imshow(white_cost_grid,
                            extent=(range_x[0], range_x[1], range_y[0], range_y[1]),
                            origin='lower', cmap=cost_cmap, vmin=0, vmax=1)
        
        # 设置标题和标签
        ax1.set_title(f'{skill_name.capitalize()} Skill - Success Rate')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        fig.colorbar(success_img, ax=ax1, label='Success Probability')
        
        ax2.set_title(f'{skill_name.capitalize()} Skill - Execution Cost')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        fig.colorbar(cost_img, ax=ax2, label='Normalized Cost')
        
        # 初始化机器人标记
        robot_plot1, = ax1.plot([], [], 'bo', markersize=10, label='Robot')
        robot_arrow1 = ax1.arrow(0, 0, 0, 0, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
        
        robot_plot2, = ax2.plot([], [], 'bo', markersize=10, label='Robot')
        robot_arrow2 = ax2.arrow(0, 0, 0, 0, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
        
        # 添加障碍物标记
        self._add_obstacles_to_plot(ax1)
        self._add_obstacles_to_plot(ax2)

        x_range_min = -2*range_radius
        x_range_max = 2*range_radius
        y_range_min = -2*range_radius
        y_range_max = 2*range_radius
        ax1.set_xlim(x_range_min, x_range_max)
        ax1.set_ylim(y_range_min, y_range_max)
        ax2.set_xlim(x_range_min, x_range_max)
        ax2.set_ylim(y_range_min, y_range_max)
        
        plt.tight_layout()
        
        def init():
            robot_plot1.set_data([], [])
            robot_plot2.set_data([], [])
            return success_img, cost_img, robot_plot1, robot_plot2
        
        def animate(i):
            obs, _ = self.env.get_observations()
            
            # 执行行动
            if i > 0:
                with torch.inference_mode():
                    # 执行动作
                    actions = policy(obs)
                    obs, _, _, _ = self.env.step(actions)

                    dist_to_goal = torch.norm(obs[0, 6:8])
                    if dist_to_goal < 0.5:
                        print(f"Task completed successfully in {i} steps.")
                        obs, _ = self.env.reset()
            
            # 获取机器人位置
            robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
            robot_rot = self.env.unwrapped.scene["robot"].data.root_quat_w[0].cpu().numpy()

            # print(self.env.unwrapped.scene["robot"].data.default_root_state)
            # print(self.env.unwrapped.scene["robot"].data)
            print(self.env.unwrapped.scene["robot"].data.root_pos_w[0])

            # 更新以机器人为中心的范围
            range_x = (robot_pos[0] - range_radius, robot_pos[0] + range_radius)
            range_y = (robot_pos[1] - range_radius, robot_pos[1] + range_radius)
        
            # 更新热图范围
            success_img.set_extent((range_x[0], range_x[1], range_y[0], range_y[1]))
            cost_img.set_extent((range_x[0], range_x[1], range_y[0], range_y[1]))

            # 更新机器人位置标记
            robot_plot1.set_data([robot_pos[0]], [robot_pos[1]])
            robot_plot2.set_data([robot_pos[0]], [robot_pos[1]])
            
            # 更新机器人朝向箭头
            from scipy.spatial.transform import Rotation as R
            r = R.from_quat([robot_rot[1], robot_rot[2], robot_rot[3], robot_rot[0]])
            heading = r.as_euler('xyz')[2]
            dx = np.cos(heading) * 0.5
            dy = np.sin(heading) * 0.5
            
            # 移除旧箭头并添加新箭头
            if hasattr(animate, 'arrow1') and animate.arrow1 in ax1.patches:
                animate.arrow1.remove()
            if hasattr(animate, 'arrow2') and animate.arrow2 in ax2.patches:
                animate.arrow2.remove()
            
            animate.arrow1 = ax1.arrow(robot_pos[0], robot_pos[1], dx, dy, 
                                       head_width=0.2, head_length=0.2, fc='blue', ec='blue')
            animate.arrow2 = ax2.arrow(robot_pos[0], robot_pos[1], dx, dy, 
                                       head_width=0.2, head_length=0.2, fc='blue', ec='blue')
            
            # 计算并更新热图
            success_grid, cost_grid = self.visualize_robot_surroundings(
                grid_size=grid_size, range_radius=range_radius,
                skill_name=skill_name, show_plot=False
            )
            
            success_img.set_array(success_grid)
            cost_img.set_array(cost_grid)
            
            return success_img, cost_img, robot_plot1, robot_plot2, animate.arrow1, animate.arrow2
        
        # 创建动画
        anim = animation.FuncAnimation(
            fig, animate, init_func=init, frames=frames, interval=interval, blit=True
        )
        
        # 保存动画
        # if save_path:
        #     anim.save(save_path, writer='pillow')
        #     print(f"Animation saved to {save_path}")
        
        plt.show()
        # plt.show(block=False)
        plt.pause(0.01)
        
        return anim

def init_sim(args_cli):
    """初始化仿真环境"""
    # 解析配置
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    
    # 创建Isaac环境
    env = gym.make(args_cli.task, cfg=env_cfg)
    
    # 包装环境
    env = RslRlVecEnvWrapper(env)
    
    # 加载策略模型
    # 这里我们假设有四种技能：walk, push, navigate, climb
    policies = {}
    
    # Walk策略
    # log_root_path = os.path.join("logs", "rsl_rl", "unitree_go2_navigation")
    # log_root_path = os.path.abspath(log_root_path)
    # resume_path = get_checkpoint_path(log_root_path, "2025-01-17_22-35-04", "model_600.pt")
    # walk_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # walk_runner.load(resume_path)
    # policies["walk"] = walk_runner.get_inference_policy(device=env.unwrapped.device)
    
    # Navigate策略 
    log_root_path = os.path.join("logs", "rsl_rl", "unitree_go2_navigation")
    log_root_path = os.path.abspath(log_root_path)
    resume_path = get_checkpoint_path(log_root_path, "nav_obstacle", "model_1499.pt")
    # resume_path = get_checkpoint_path(log_root_path, "terrain_adaptation", "model_550.pt")
    nav_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    nav_runner.load(resume_path)
    policies["navigate"] = nav_runner.get_inference_policy(device=env.unwrapped.device)
    
    # 将策略添加到环境对象中，方便后续访问
    env.policies = policies
    
    return env, policies

def collect_dataset(env, policies):
    """收集数据集"""
    # 初始化数据收集器
    data_collector = SkillEffectDataCollector(env, policies)
    
    # 为每种技能收集数据
    # data_collector.collect_data_for_skill("walk", num_trials=200)
    data_collector.collect_data_for_skill("navigate", num_trials=10)
    
    # 保存完整数据集
    data_collector.save_data("complete")
    
    return data_collector

def train_model(data_path):
    """训练效果预测模型"""
    # 初始化模型训练器
    model_trainer = SkillEffectModelTrainer()
    
    # 训练模型
    model = model_trainer.train_model(data_path, batch_size=64, epochs=100)
    
    # 保存模型
    model_trainer.save_model()
    
    return model_trainer

def test_and_visualize_model(env, model_trainer, policies):
    """测试和可视化效果模型"""
    # 初始化可视化器
    visualizer = SkillEffectVisualizer(env, model_trainer)
    
    # 可视化各种技能在不同目标点的效果
    # visualizer.visualize_robot_surroundings(skill_name="walk", save_path="viz/walk_effect.png")
    # visualizer.visualize_robot_surroundings(skill_name="navigate", save_path="viz/navigate_effect.png")
    
    # 比较不同技能对同一目标的效果
    # target_position = np.array([2.0, 2.0])
    # visualizer.visualize_skill_comparison(target_position, save_path="viz/skill_comparison.png")
    
    # 创建动态可视化
    visualizer.create_animation(skill_name="navigate", frames=50, save_path="viz/walk_animation.gif")

    return visualizer

def main():
    """主函数"""
    
    # 初始化仿真环境
    env, policies = init_sim(args_cli)
    
    # 根据模式执行操作
    if args.mode in ["collect", "all"]:
        print("Collecting dataset...")
        data_collector = collect_dataset(env, policies)
    
    if args.mode in ["train", "all"]:
        print("Training model...")
        model_trainer = train_model(args.data_path)
    else:
        # 加载预训练模型
        print("Loading pre-trained model...")
        model_trainer = SkillEffectModelTrainer()
        if args.model_path:
            model_trainer.load_model(args.model_path)
        else:
            # 找到最新的模型文件
            model_files = [f for f in os.listdir("models/skill_effect") if f.endswith('.pt')]
            if model_files:
                latest_model = sorted(model_files)[-1]
                model_path = os.path.join("models/skill_effect", latest_model)
                model_trainer.load_model(model_path)
            else:
                raise ValueError("No model file found. Please specify --model_path or run with --mode train.")
    
    if args.mode in ["test", "all"]:
        print("Testing and visualizing model...")
        visualizer = test_and_visualize_model(env, model_trainer, policies)
    
    # 关闭仿真环境
    env.close()
    
    # 关闭Omniverse应用
    simulation_app.close()

if __name__ == "__main__":
    main()