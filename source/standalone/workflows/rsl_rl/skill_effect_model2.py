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

# 添加RSL-RL和AppLauncher参数
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动Omniverse应用
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 解析参数
args = parser.parse_args()

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
from omni.isaac.lab.utils.math import subtract_frame_transforms
from scipy.spatial.transform import Rotation as R

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

        # 检查并处理输出中的NaN
        if torch.isnan(success_prob).any() or torch.isnan(cost).any():
            print("Warning: Model output contains NaN values, replacing with the maximum cost")
            success_prob = torch.nan_to_num(success_prob, nan=0.0)
            cost = torch.nan_to_num(cost, nan=1.0)

        return success_prob, cost

class SkillEffectModelTrainer:
    """技能效果模型训练器"""
    
    def __init__(self, model_save_dir="models/skill_effect"):
        self.model_save_dir = model_save_dir
        os.makedirs(model_save_dir, exist_ok=True)
        self.model = None
        self.scaler = None
        
    # def train_model(self, dataset_path, batch_size=64, epochs=100, lr=0.001, test_size=0.2, device='cuda'):
    #     """训练模型"""
    #     print(f"Loading dataset from {dataset_path}")
    #     with open(dataset_path, 'rb') as f:
    #         data = pickle.load(f)
        
    #     features = data['features']
    #     success_labels = data['success_labels']
    #     cost_labels = data['cost_labels']

    #     # 检查并清理NaN值
    #     feature_nan = np.isnan(features).any(axis=1)
    #     success_nan = np.isnan(success_labels)
    #     cost_nan = np.isnan(cost_labels)

    #     # 合并所有NaN索引（使用逻辑OR操作）
    #     nan_indices = feature_nan | success_nan | cost_nan
    #     # if nan_indices.any():
    #     #     print(f"Warning: Found {nan_indices.sum()} samples with NaN values, removing them")
    #     #     features = features[~nan_indices]
    #     #     success_labels = success_labels[~nan_indices]
    #     #     cost_labels = cost_labels[~nan_indices]
        
    #     print(f"Dataset size: {len(features)}")
    #     print(f"Features shape: {features.shape}")
    #     print(f"Success rate distribution: {np.mean(success_labels):.4f}")
    #     print(f"Cost mean: {np.mean(cost_labels):.4f}, std: {np.std(cost_labels):.4f}")
        
    #     # 拆分训练集和测试集
    #     X_train, X_test, y_success_train, y_success_test, y_cost_train, y_cost_test = train_test_split(
    #         features, success_labels, cost_labels, test_size=test_size, random_state=42
    #     )
        
    #     # 创建数据加载器
    #     train_dataset = SkillEffectDataset(X_train, y_success_train, y_cost_train)
    #     test_dataset = SkillEffectDataset(X_test, y_success_test, y_cost_test)
        
    #     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    #     test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
    #     # 创建模型
    #     input_dim = features.shape[1]
    #     # print(f"Input dimension: {input_dim}")

    #     model = SkillEffectModel(input_dim=input_dim)
    #     model.to(device)
        
    #     # 定义优化器和损失函数
    #     optimizer = optim.Adam(model.parameters(), lr=lr)
    #     success_criterion = nn.BCELoss()
    #     cost_criterion = nn.MSELoss()
        
    #     # 训练循环
    #     best_val_loss = float('inf')
    #     patience = 10
    #     patience_counter = 0
        
    #     for epoch in range(epochs):
    #         model.train()
    #         train_success_loss = 0
    #         train_cost_loss = 0
            
    #         for batch_features, batch_success, batch_cost in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
    #             batch_features = batch_features.to(device)
    #             batch_success = batch_success.to(device).unsqueeze(1)
    #             batch_cost = batch_cost.to(device).unsqueeze(1)
                
    #             # 梯度清零
    #             optimizer.zero_grad()
                
    #             # 前向传播
    #             success_pred, cost_pred = model(batch_features)
                
    #             # 计算损失
    #             success_loss = success_criterion(success_pred, batch_success)
    #             cost_loss = cost_criterion(cost_pred, batch_cost)
                
    #             # 总损失
    #             loss = success_loss + cost_loss
                
    #             # 反向传播
    #             loss.backward()
                
    #             # 更新权重
    #             optimizer.step()
                
    #             train_success_loss += success_loss.item()
    #             train_cost_loss += cost_loss.item()
            
    #         # 验证
    #         model.eval()
    #         val_success_loss = 0
    #         val_cost_loss = 0
    #         success_preds = []
    #         success_true = []
    #         cost_preds = []
    #         cost_true = []
            
    #         with torch.no_grad():
    #             for batch_features, batch_success, batch_cost in test_loader:
    #                 batch_features = batch_features.to(device)
    #                 batch_success = batch_success.to(device).unsqueeze(1)
    #                 batch_cost = batch_cost.to(device).unsqueeze(1)
                    
    #                 success_pred, cost_pred = model(batch_features)

    #                 # print("11")
    #                 # 检查是否有NaN输出
    #                 if torch.isnan(success_pred).any():
    #                     print(f"Warning: NaN values detected in success predictions: {torch.isnan(success_pred).sum().item()} NaNs")
                        
    #                 if torch.isnan(cost_pred).any():
    #                     print(f"Warning: NaN values detected in cost predictions: {torch.isnan(cost_pred).sum().item()} NaNs")
                    
    #                 val_success_loss += success_criterion(success_pred, batch_success).item()
    #                 val_cost_loss += cost_criterion(cost_pred, batch_cost).item()
                    
    #                 # 收集预测和真实值
    #                 success_preds.extend(success_pred.cpu().numpy())
    #                 success_true.extend(batch_success.cpu().numpy())
    #                 cost_preds.extend(cost_pred.cpu().numpy())
    #                 cost_true.extend(batch_cost.cpu().numpy())
            
    #         # 计算验证指标 - 全部使用回归指标
    #         from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    #         success_mse = mean_squared_error(success_true, success_preds)
    #         success_mae = mean_absolute_error(success_true, success_preds) 
    #         success_r2 = r2_score(success_true, success_preds)
            
    #         cost_mse = mean_squared_error(cost_true, cost_preds)
    #         cost_mae = mean_absolute_error(cost_true, cost_preds)
    #         cost_r2 = r2_score(cost_true, cost_preds)
            
    #         # 输出训练信息
    #         print(f"Epoch {epoch+1}/{epochs}")
    #         print(f"  Train: success_loss: {train_success_loss/len(train_loader):.4f}, cost_loss: {train_cost_loss/len(train_loader):.4f}")
    #         print(f"  Val: success_loss: {val_success_loss/len(test_loader):.4f}, cost_loss: {val_cost_loss/len(test_loader):.4f}")
    #         print(f"  Success MSE: {success_mse:.4f}, MAE: {success_mae:.4f}, R²: {success_r2:.4f}")
    #         print(f"  Cost MSE: {cost_mse:.4f}, MAE: {cost_mae:.4f}, R²: {cost_r2:.4f}")
    
    #         # 早停
    #         val_loss = val_success_loss/len(test_loader) + val_cost_loss/len(test_loader)
    #         if val_loss < best_val_loss:
    #             best_val_loss = val_loss
    #             patience_counter = 0
    #             # 保存最佳模型
    #             timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #             model_path = os.path.join(self.model_save_dir, f"skill_effect_model_{timestamp}.pt")
    #             torch.save(model.state_dict(), model_path)
    #             print(f"Model saved to {model_path}")
    #         else:
    #             patience_counter += 1
            
    #         if patience_counter >= patience:
    #             print(f"Early stopping at epoch {epoch+1}")
    #             break
        
    #     # 加载最佳模型
    #     model_files = [f for f in os.listdir(self.model_save_dir) if f.endswith('.pt')]
    #     if model_files:
    #         latest_model = sorted(model_files)[-1]
    #         model_path = os.path.join(self.model_save_dir, latest_model)
    #         model.load_state_dict(torch.load(model_path))
    #         print(f"Loaded best model from {model_path}")
        
    #     self.model = model
    #     return model

    def train_model(self, dataset_path, batch_size=64, epochs=100, lr=0.001, test_size=0.2, device='cuda'):
        """分阶段训练模型"""
        print(f"Loading dataset from {dataset_path}")
        with open(dataset_path, 'rb') as f:
            data = pickle.load(f)
        
        features = data['features']
        success_labels = data['success_labels']
        cost_labels = data['cost_labels']

        # 检查并清理NaN值
        feature_nan = np.isnan(features).any(axis=1)
        success_nan = np.isnan(success_labels)
        cost_nan = np.isnan(cost_labels)

        # 合并所有NaN索引（使用逻辑OR操作）
        nan_indices = feature_nan | success_nan | cost_nan
        if nan_indices.any():
            print(f"Warning: Found {nan_indices.sum()} samples with NaN values, removing them")
            features = features[~nan_indices]
            success_labels = success_labels[~nan_indices]
            cost_labels = cost_labels[~nan_indices]

        # 检查并清理无穷值和极端值
        inf_mask = np.isinf(features).any(axis=1)
        # 也可以设置一个合理的阈值来过滤极端值
        extreme_mask = np.abs(features).max(axis=1) > 1e6
        invalid_mask = inf_mask | extreme_mask
        
        if invalid_mask.any():
            print(f"Warning: Found {invalid_mask.sum()} samples with infinity or extreme values, removing them")
            features = features[~invalid_mask]
            success_labels = success_labels[~invalid_mask]
            cost_labels = cost_labels[~invalid_mask]
        
        # 数据清理和打印与原始代码相同
        print(f"Dataset size: {len(features)}")
        print(f"Features shape: {features.shape}")
        print(f"Success rate distribution: {np.mean(success_labels):.4f}")
        print(f"Cost mean: {np.mean(cost_labels):.4f}, std: {np.std(cost_labels):.4f}")
        
        # 拆分训练集和测试集
        X_train, X_test, y_success_train, y_success_test, y_cost_train, y_cost_test = train_test_split(
            features, success_labels, cost_labels, test_size=test_size, random_state=42
        )

        # 添加数据标准化
        # from sklearn.preprocessing import StandardScaler
        # scaler = StandardScaler()
        # features_scaled = scaler.fit_transform(features)
        # self.scaler = scaler  # 保存scaler以便在预测时使用
        
        # # 使用标准化后的特征
        # X_train, X_test, y_success_train, y_success_test, y_cost_train, y_cost_test = train_test_split(
        #     features_scaled, success_labels, cost_labels, test_size=test_size, random_state=42
        # )
        
        # 创建数据加载器
        train_dataset = SkillEffectDataset(X_train, y_success_train, y_cost_train)
        test_dataset = SkillEffectDataset(X_test, y_success_test, y_cost_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # 创建模型
        input_dim = features.shape[1]
        model = SkillEffectModel(input_dim=input_dim)
        model.to(device)
        
        # 定义损失函数
        success_criterion = nn.BCELoss()
        cost_criterion = nn.MSELoss()
        
        # ==== 阶段1: 训练成功率预测部分 ====
        print("Phase 1: Training success prediction head and shared layers...")
        
        # 冻结代价预测部分
        for param in model.cost_head.parameters():
            param.requires_grad = False
        
        # 优化器只优化未冻结的部分
        optimizer_success = optim.Adam([
            {'params': model.shared_layers.parameters()},
            {'params': model.success_head.parameters()}
        ], lr=lr)
        
        best_success_val_loss = float('inf')
        success_phase_epochs = min(30, epochs // 3)  # 总epochs的1/3用于success训练
        
        for epoch in range(success_phase_epochs):
            # 训练循环
            model.train()
            train_success_loss = 0
            
            for batch_features, batch_success, _ in tqdm(train_loader, desc=f"Success Phase - Epoch {epoch+1}/{success_phase_epochs}"):
                batch_features = batch_features.to(device)
                batch_success = batch_success.to(device).unsqueeze(1)
                
                optimizer_success.zero_grad()
                success_pred, _ = model(batch_features)
                success_loss = success_criterion(success_pred, batch_success)
                success_loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 添加梯度裁剪
                optimizer_success.step()
                
                train_success_loss += success_loss.item()
            
            # 验证
            model.eval()
            val_success_loss = 0
            success_preds = []
            success_true = []
            
            with torch.no_grad():
                for batch_features, batch_success, _ in test_loader:
                    batch_features = batch_features.to(device)
                    batch_success = batch_success.to(device).unsqueeze(1)
                    
                    success_pred, _ = model(batch_features)
                    val_success_loss += success_criterion(success_pred, batch_success).item()
                    
                    success_preds.extend(success_pred.cpu().numpy())
                    success_true.extend(batch_success.cpu().numpy())
            
            # 计算验证指标
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            success_mse = mean_squared_error(success_true, success_preds)
            success_mae = mean_absolute_error(success_true, success_preds) 
            success_r2 = r2_score(success_true, success_preds)
            
            # 输出训练信息
            print(f"  Success Phase - Epoch {epoch+1}/{success_phase_epochs}")
            print(f"  Train success_loss: {train_success_loss/len(train_loader):.4f}")
            print(f"  Val success_loss: {val_success_loss/len(test_loader):.4f}")
            print(f"  Success MSE: {success_mse:.4f}, MAE: {success_mae:.4f}, R²: {success_r2:.4f}")
            
            # 保存最佳success模型
            if val_success_loss < best_success_val_loss:
                best_success_val_loss = val_success_loss
                success_model_path = os.path.join(self.model_save_dir, f"skill_effect_model_success_phase.pt")
                torch.save(model.state_dict(), success_model_path)
                print(f"  Success phase model saved to {success_model_path}")
        
        # 加载最佳success模型
        model.load_state_dict(torch.load(success_model_path))
        print(f"Loaded best success phase model")
        
        # ==== 阶段2: 训练代价预测部分 ====
        print("Phase 2: Training cost prediction head while freezing success head...")
        
        # 解冻代价预测部分，冻结成功率预测部分
        for param in model.cost_head.parameters():
            param.requires_grad = True
        for param in model.success_head.parameters():
            param.requires_grad = False
        
        # 优化器只优化代价预测部分和共享层
        optimizer_cost = optim.Adam([
            {'params': model.shared_layers.parameters()},
            {'params': model.cost_head.parameters()}
        ], lr=lr)
        
        best_cost_val_loss = float('inf')
        cost_phase_epochs = min(30, epochs // 3)  # 总epochs的1/3用于cost训练
        
        for epoch in range(cost_phase_epochs):
            # 训练循环
            model.train()
            train_cost_loss = 0
            
            for batch_features, _, batch_cost in tqdm(train_loader, desc=f"Cost Phase - Epoch {epoch+1}/{cost_phase_epochs}"):
                batch_features = batch_features.to(device)
                batch_cost = batch_cost.to(device).unsqueeze(1)
                
                optimizer_cost.zero_grad()
                _, cost_pred = model(batch_features)
                cost_loss = cost_criterion(cost_pred, batch_cost)
                cost_loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 添加梯度裁剪
                optimizer_cost.step()
                
                train_cost_loss += cost_loss.item()
            
            # 验证
            model.eval()
            val_cost_loss = 0
            cost_preds = []
            cost_true = []
            
            with torch.no_grad():
                for batch_features, _, batch_cost in test_loader:
                    batch_features = batch_features.to(device)
                    batch_cost = batch_cost.to(device).unsqueeze(1)
                    
                    _, cost_pred = model(batch_features)
                    val_cost_loss += cost_criterion(cost_pred, batch_cost).item()
                    
                    cost_preds.extend(cost_pred.cpu().numpy())
                    cost_true.extend(batch_cost.cpu().numpy())
            
            # 计算验证指标
            cost_mse = mean_squared_error(cost_true, cost_preds)
            cost_mae = mean_absolute_error(cost_true, cost_preds)
            cost_r2 = r2_score(cost_true, cost_preds)
            
            # 输出训练信息
            print(f"  Cost Phase - Epoch {epoch+1}/{cost_phase_epochs}")
            print(f"  Train cost_loss: {train_cost_loss/len(train_loader):.4f}")
            print(f"  Val cost_loss: {val_cost_loss/len(test_loader):.4f}")
            print(f"  Cost MSE: {cost_mse:.4f}, MAE: {cost_mae:.4f}, R²: {cost_r2:.4f}")
            
            # 保存最佳cost模型
            if val_cost_loss < best_cost_val_loss:
                best_cost_val_loss = val_cost_loss
                cost_model_path = os.path.join(self.model_save_dir, f"skill_effect_model_cost_phase.pt")
                torch.save(model.state_dict(), cost_model_path)
                print(f"  Cost phase model saved to {cost_model_path}")
        
        # 加载最佳cost模型
        model.load_state_dict(torch.load(cost_model_path))
        print(f"Loaded best cost phase model")
        
        # ==== 阶段3: 微调整个模型 ====
        print("Phase 3: Fine-tuning all parameters together...")
        
        # 解冻所有参数
        for param in model.parameters():
            param.requires_grad = True
        
        # 使用较小的学习率微调整个模型
        optimizer_finetune = optim.Adam(model.parameters(), lr=lr/5)
        
        # 训练循环
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        finetune_epochs = epochs - success_phase_epochs - cost_phase_epochs
        
        for epoch in range(finetune_epochs):
            model.train()
            train_success_loss = 0
            train_cost_loss = 0
            
            for batch_features, batch_success, batch_cost in tqdm(train_loader, desc=f"Fine-tuning - Epoch {epoch+1}/{finetune_epochs}"):
                batch_features = batch_features.to(device)
                batch_success = batch_success.to(device).unsqueeze(1)
                batch_cost = batch_cost.to(device).unsqueeze(1)
                
                # 梯度清零
                optimizer_finetune.zero_grad()
                
                # 前向传播
                success_pred, cost_pred = model(batch_features)
                
                # 计算损失 - 平衡两个任务的权重
                success_loss = success_criterion(success_pred, batch_success)
                cost_loss = cost_criterion(cost_pred, batch_cost)
                
                # 总损失 - 可以调整权重
                success_weight = 1.0
                cost_weight = 1.0
                loss = success_weight * success_loss + cost_weight * cost_loss
                
                # 反向传播
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 添加梯度裁剪
                
                # 更新权重
                optimizer_finetune.step()
                
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
            
            # 计算验证指标
            success_mse = mean_squared_error(success_true, success_preds)
            success_mae = mean_absolute_error(success_true, success_preds) 
            success_r2 = r2_score(success_true, success_preds)
            
            cost_mse = mean_squared_error(cost_true, cost_preds)
            cost_mae = mean_absolute_error(cost_true, cost_preds)
            cost_r2 = r2_score(cost_true, cost_preds)
            
            # 输出训练信息
            print(f"Fine-tuning - Epoch {epoch+1}/{finetune_epochs}")
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
        model_files = [f for f in os.listdir(self.model_save_dir) if f.endswith('.pt') and not f.endswith('_phase.pt')]
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
        
    def load_model(self, model_path, input_dim=None, device='cuda'):
        """加载已训练的模型"""
        # 先检查模型的输入维度
        state_dict = torch.load(model_path)
        
        # 从共享层的权重形状获取输入维度
        if input_dim is None:
            input_dim = state_dict['shared_layers.0.weight'].shape[1]
            print(f"Detected input dimension from saved model: {input_dim}")
        
        model = SkillEffectModel(input_dim=input_dim)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        self.model = model
        print(f"Model loaded from {model_path} with input dimension {input_dim}")
        return model
    
    def predict(self, features):
        """预测技能成功率和代价"""
        if self.model is None:
            raise ValueError("Model not loaded or trained")
        
        with torch.no_grad():
            features_tensor = torch.tensor(features, dtype=torch.float32).to(next(self.model.parameters()).device)
            success_prob, cost = self.model(features_tensor)
        
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
        
    def collect_data_for_skill(self, skill_name, num_trials=500, max_steps=50):
        """为特定技能收集数据 - 支持多环境并行收集"""
        print(f"Collecting data for skill: {skill_name}")
        policy = self.policies[skill_name]
        
        num_envs = self.env.num_envs
        print(f"Using {num_envs} parallel environments for data collection")
        
        # 计算每个环境需要执行的试验次数
        trials_per_env = num_trials // num_envs
        remaining_trials = num_trials % num_envs
        
        # 每个环境的数据收集计数
        env_trial_counts = [trials_per_env] * num_envs
        # 将剩余的试验分配给前几个环境
        for i in range(remaining_trials):
            env_trial_counts[i] += 1
        
        # 总计数
        total_trials_done = 0
        
        # 分批次收集数据
        while total_trials_done < num_trials:
            # 确定本批次每个环境的试验次数
            batch_size = min(100, num_trials - total_trials_done)
            env_batch_sizes = [min(env_trial_counts[i], batch_size) for i in range(num_envs)]
            
            # 重置所有环境
            with torch.inference_mode():
                obs, _ = self.env.reset()
            
            # 为每个环境生成目标位置
            target_positions = []
            for i in range(num_envs):
                if env_trial_counts[i] > 0:  # 如果这个环境还有试验要做
                    target_x = np.random.uniform(-3, 3)
                    target_y = np.random.uniform(-3, 3)
                    target_x = 2.0
                    target_y = 2.0
                    
                    if skill_name == "push":
                        target_position = torch.tensor([target_x, target_y], device=self.env.unwrapped.device)
                    elif skill_name == "climb":
                        target_z = np.random.uniform(0.2, 0.6)
                        target_position = torch.tensor([target_x, target_y, target_z], device=self.env.unwrapped.device)
                    else:  # walk or navigate
                        target_position = torch.tensor([target_x, target_y], device=self.env.unwrapped.device)
                    
                    target_positions.append(target_position)
                else:
                    # 这个环境已经完成了所有试验，使用占位目标
                    target_positions.append(None)
            
            # 设置每个环境的初始观测和目标
            active_envs = []  # 活跃环境索引列表
            for i in range(num_envs):
                if env_trial_counts[i] > 0:
                    active_envs.append(i)
                    # 设置目标相对位置
                    robot_state = self.env.unwrapped.scene["robot"].data.root_state_w
                    from omni.isaac.lab.utils.math import subtract_frame_transforms
                    goal, _ = subtract_frame_transforms(
                        robot_state[i:i+1, :3], 
                        robot_state[i:i+1, 3:7], 
                        target_positions[i][:3] if target_positions[i].shape[0] > 2 else 
                        torch.cat([target_positions[i], torch.zeros(1, device=target_positions[i].device)])
                    )
                    
                    # 更新观测中的目标位置
                    # with torch.inference_mode():
                    #     obs[i, 6:8] = goal[0, :2]
            
            # 提取初始特征
            env_features = []
            for i in active_envs:
                features = self._extract_features(obs[i:i+1], target_positions[i], env_idx=i)
                env_features.append(features)
            
            # 执行技能
            steps_taken = [0] * num_envs
            task_finished = [False] * num_envs
            env_success_rates = [0.0] * num_envs
            
            with torch.inference_mode():
                for step in range(max_steps):
                    # 获取动作
                    actions = policy(obs)
                    
                    # 执行动作
                    obs, _, _, _ = self.env.step(actions)
                    
                    # 更新每个环境的状态
                    for i in active_envs:
                        if not task_finished[i]:
                            steps_taken[i] += 1
                            
                            # 检查是否完成任务
                            dist_to_goal = torch.norm(obs[i, 6:8])
                            if dist_to_goal < 0.25:
                                task_finished[i] = True
                                env_success_rates[i] = 1.0
                                print(f"Env {i}: Task completed in {steps_taken[i]} steps.")
                    
                    # 如果所有活跃环境都完成了任务，提前结束
                    if all(task_finished[i] for i in active_envs):
                        break
            
            # 处理未完成的环境
            for i in active_envs:
                if not task_finished[i]:
                    dist_to_goal = torch.norm(obs[i, 6:8])
                    if dist_to_goal > 2.0:
                        env_success_rates[i] = 0.0
                    else:
                        # env_success_rates[i] = np.exp(-2 * (dist_to_goal.cpu().numpy() - 0.25))
                        env_success_rates[i] = 0.0
                    print(f"Env {i}: Task failed. Distance: {dist_to_goal:.2f}, Success rate: {env_success_rates[i]:.2f}")
            
            # 收集结果
            for i in active_envs:
                cost = steps_taken[i] / max_steps
                
                # 存储数据
                self.features.append(env_features[i])
                self.success_labels.append(float(env_success_rates[i]))
                self.cost_labels.append(cost)
                self.skill_types.append(skill_name)
                self.targets.append(target_positions[i].cpu().numpy())
                
                # 更新试验计数
                env_trial_counts[i] -= 1
                total_trials_done += 1
            
            # 每100个样本保存一次数据
            if total_trials_done % 100 == 0:
                self.save_data(f"{skill_name}_partial_{total_trials_done}")
        
        # 最终保存数据
        self.save_data(f"{skill_name}_complete_{num_trials}")
        
    def _extract_features(self, obs, target_position, env_idx=0):
        """
        从观测中提取特征
        
        Args:
            obs: 单个环境的观测
            target_position: 目标位置
            env_idx: 环境索引
        """
        # 获取高度扫描数据 - 确保使用正确的环境索引
        # lidar_data = self.env.unwrapped.scene.sensors["height_scanner_wide"].data.ray_hits_w[env_idx].cpu().numpy()
        # lidar_data = self.env.unwrapped.scene.sensors["height_scanner"].data.ray_hits_w[env_idx].cpu().numpy()
        
        lidar_pos = self.env.unwrapped.scene.sensors["height_scanner_wide"].data.pos_w[env_idx]
        lidar_data = self.env.unwrapped.scene.sensors["height_scanner_wide"].data.ray_hits_w[env_idx]
        
        # print("lidar_pos shape:", lidar_pos.shape)  # 应为 [N, 3]
        # print("lidar_data shape:", lidar_data.shape)  # 应为 [M, 3] 或 [B, M, 3]

        offset = 0.5
        lidar_data_offset = lidar_pos[2] - lidar_data[..., 2] - offset
        lidar_data_offset = lidar_data_offset.cpu().numpy()
        lidar_data_offset = lidar_data_offset.flatten()
        # print(lidar_data)
        # lidar_data = lidar_data[2::3]

        # print("lidar: ",lidar_data_offset)
        # print("obs: ",obs[0,10:])
        
        # 提取观测特征
        features = obs[0, 6:].cpu().numpy()  # 获取第一个（也是唯一的）元素，因为obs已经是切片
        # print("obs: ",obs[0,10:])

        # 获取机器人位置 - 使用正确的环境索引
        robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[env_idx].cpu().numpy()
        
        # 获取高度特征
        height_features = lidar_data_offset
        
        # 计算目标相对位置 (显式特征)
        # target_rel_pos = target_position.cpu().numpy()[:2] - robot_pos[:2]

        target_rel_pos = obs[0, 6:8].cpu().numpy()
        
        # 将所有特征合并
        combined_features = np.concatenate([
            target_rel_pos,     # 目标相对位置 [x, y]
            height_features     # 高度扫描特征
        ])
        
        # return features
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
                # if skill_name == "walk":
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

                # print(obs[:, 6:8])
                # print(torch.norm(obs[:, 6:8], dim=1))
                # dist_to_goal = torch.norm(obs[0, 6:8])
                dist_to_goal = torch.norm(obs[:, 6:8], dim=1)
                task_finished = dist_to_goal < 0.25

                print("Distance to goal:", dist_to_goal)

                
        if task_finished:
            print(f"Task completed successfully in {steps} steps.")
            success_rate = 1.0
        else:
            print(f"Task failed.")
            if dist_to_goal > 2.0:
                success_rate = 0.0
            else:
                success_rate = np.exp(-2 * (dist_to_goal.cpu().numpy() - 0.25))
        
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
        # lidar_data = self.env.unwrapped.scene.sensors["height_scanner_wide"].data.ray_hits_w[0].cpu().numpy()
        # lidar_data = self.env.unwrapped.scene.sensors["height_scanner"].data.ray_hits_w[0].cpu().numpy()

        lidar_pos = self.env.unwrapped.scene.sensors["height_scanner_wide"].data.pos_w[0]
        lidar_data = self.env.unwrapped.scene.sensors["height_scanner_wide"].data.ray_hits_w[0]

        offset = 0.5
        lidar_data_offset = lidar_pos[2] - lidar_data[..., 2] - offset
        lidar_data_offset = lidar_data_offset.cpu().numpy()
        lidar_data_offset = lidar_data_offset.flatten()

        # 获取机器人位置
        robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
        robot_rot = self.env.unwrapped.scene["robot"].data.root_quat_w[0].cpu().numpy()
        robot_state = self.env.unwrapped.scene["robot"].data.root_state_w

        # 计算机器人的朝向角
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat([robot_rot[1], robot_rot[2], robot_rot[3], robot_rot[0]])  # wxyz to xyzw
        heading = r.as_euler('xyz')[2]  # 获取yaw角

        rotation_matrix = np.array([
            [np.cos(heading), np.sin(heading)],
            [-np.sin(heading), np.cos(heading)]
        ])
        
        # 创建以机器人为中心的网格（而不是固定的绝对坐标）
        range_x = (robot_pos[0] - range_radius, robot_pos[0] + range_radius)
        range_y = (robot_pos[1] - range_radius, robot_pos[1] + range_radius)
        
        x = np.arange(range_x[0], range_x[1], grid_size)
        y = np.arange(range_y[0], range_y[1], grid_size)
        X, Y = np.meshgrid(x, y)

        # 创建以机器人为中心的网格（相对坐标系）
        # x = np.arange(-range_radius, range_radius, grid_size)
        # y = np.arange(-range_radius, range_radius, grid_size)
        # X, Y = np.meshgrid(x, y)
        
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

                feature = obs[0, 6:].cpu().numpy()  # 取第一个环境的观测
                # feature[:2] = target_rel_pos.flatten()  # 更新目标相对位置
                feature[:2] = point  # 高度扫描数据

                # 获取高度特征
                height_features = lidar_data_offset

                target_pos = torch.tensor([point[0], point[1], 0.0], device=self.env.unwrapped.device, dtype=torch.float32)
                target_rel_pos, _ = subtract_frame_transforms(robot_state[:, :3], robot_state[:, 3:7], target_pos)
                target_rel_pos = target_rel_pos.cpu().numpy().flatten()[:2]  # 只取x, y坐标

                # target_rel_pos = point
                # print(target_rel_pos)

                # 将所有特征合并
                combined_features = np.concatenate([
                    target_rel_pos,     # 目标相对位置 [x, y]
                    height_features     # 高度扫描特征
                ])

                # print("Feature shape:", feature.shape)
                
                batch_features.append(combined_features)

                # feature[:2] = target_rel_pos  # 更新目标相对位置
                # batch_features.append(feature)
            
            # 进行预测
            batch_features = np.array(batch_features)
            batch_success_probs, batch_costs = self.model.predict(batch_features)
            
            # print("batch_success_probs: ", batch_success_probs)
            # print("batch_costs: ", batch_costs)

            # 更新结果
            success_probs[i:i+len(batch_features)] = batch_success_probs.flatten()
            costs[i:i+len(batch_features)] = batch_costs.flatten()
        
        # 重塑结果为网格形状
        success_probs_grid = success_probs.reshape(X.shape)
        costs_grid = costs.reshape(X.shape)
        
        # 计算数据的实际分布范围，以优化可视化效果
        success_min = np.nanmin(success_probs_grid)
        success_max = np.nanmax(success_probs_grid)
        cost_min = np.nanmin(costs_grid)
        cost_max = np.nanmax(costs_grid)
        
        print(f"Success probability range: {success_min:.4f} to {success_max:.4f}")
        print(f"Cost range: {cost_min:.4f} to {cost_max:.4f}")
        
        # 动态设置可视化范围，但确保有最小区间以显示差异
        success_range = max(success_max - success_min, 0.2)  # 确保至少有0.2的范围
        cost_range = max(cost_max - cost_min, 0.2)           # 确保至少有0.2的范围
        
        # 计算更合适的可视化范围
        success_vmin = max(0, success_min - 0.1 * success_range)  # 向下扩展10%
        success_vmax = min(1, success_max + 0.1 * success_range)  # 向上扩展10%
        cost_vmin = max(0, cost_min - 0.1 * cost_range)           # 向下扩展10%
        cost_vmax = min(1, cost_max + 0.1 * cost_range)           # 向上扩展10%
        
        # 创建图表 - 现在是3个子图：高度图、成功率和代价
        fig, (ax_height, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 7))
        
        # 绘制高度地图
        self.visualize_height_map(ax_height, grid_size, range_radius)
        
        # 成功率热图 - 使用新的范围参数
        success_cmap = LinearSegmentedColormap.from_list('success_cmap', ['red', 'yellow', 'green'])
        success_img = ax1.imshow(success_probs_grid, extent=(range_x[0], range_x[1], range_y[0], range_y[1]), 
                                origin='lower', cmap=success_cmap, vmin=success_vmin, vmax=success_vmax)
        ax1.set_title(f'{skill_name.capitalize()} Skill - Success Rate\n(Range: {success_vmin:.2f} to {success_vmax:.2f})')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        fig.colorbar(success_img, ax=ax1, label='Success Probability')
        
        ax1.text(0.5, -0.15, f"Min: {success_min:.4f}, Max: {success_max:.4f}",
                horizontalalignment='center', verticalalignment='center',
                transform=ax1.transAxes, fontsize=10, fontweight='bold')

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
                            origin='lower', cmap=cost_cmap, vmin=cost_vmin, vmax=cost_vmax)
        ax2.set_title(f'{skill_name.capitalize()} Skill - Execution Cost\n(Range: {cost_vmin:.2f} to {cost_vmax:.2f})')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        fig.colorbar(cost_img, ax=ax2, label='Normalized Cost')

        # 在图表下方添加最大最小值文本信息
        ax2.text(0.5, -0.15, f"Min: {cost_min:.4f}, Max: {cost_max:.4f}",
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes, fontsize=10, fontweight='bold')
        
        # 标记机器人位置
        ax2.plot(robot_pos[0], robot_pos[1], 'bo', markersize=10, label='Robot')
        ax2.legend()
        ax2.arrow(robot_pos[0], robot_pos[1], dx, dy, head_width=0.2, head_length=0.2, fc='blue', ec='blue')

        
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

    def visualize_height_map(self, ax, grid_size=0.1, range_radius=5.0):
        """在给定的轴上绘制机器人周围的高度地图"""
        # 获取高度扫描数据
        lidar_data = self.env.unwrapped.scene.sensors["height_scanner_wide"].data.ray_hits_w[0].cpu().numpy()
        lidar_raw = lidar_data.reshape(-1, 3)  # 假设每3个值为一组[x,y,z]坐标
        
        # 提取坐标
        x_points = lidar_raw[:, 0]
        y_points = lidar_raw[:, 1]
        height_values = lidar_raw[:, 2] #- 0.52  # 调整高度值
        
        # 获取机器人位置和朝向
        robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
        robot_rot = self.env.unwrapped.scene["robot"].data.root_quat_w[0].cpu().numpy()
        
        # 创建网格
        range_x = (robot_pos[0] - range_radius, robot_pos[0] + range_radius)
        range_y = (robot_pos[1] - range_radius, robot_pos[1] + range_radius)
        
        # 使用散点图绘制高度数据，颜色表示高度
        scatter = ax.scatter(x_points, y_points, c=height_values, cmap='terrain', 
                            s=10, alpha=0.8, vmin=-0.5, vmax=0.5)
        
        # 标记机器人位置
        ax.plot(robot_pos[0], robot_pos[1], 'ro', markersize=10, label='Robot')
        
        # 绘制朝向箭头
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat([robot_rot[1], robot_rot[2], robot_rot[3], robot_rot[0]])  # wxyz to xyzw
        heading = r.as_euler('xyz')[2]
        dx = np.cos(heading) * 0.5
        dy = np.sin(heading) * 0.5
        ax.arrow(robot_pos[0], robot_pos[1], dx, dy, head_width=0.2, head_length=0.2, 
                fc='red', ec='red', label='Heading')
        
        # 设置图表属性
        ax.set_xlim(range_x)
        ax.set_ylim(range_y)
        ax.set_title('Terrain Height Map')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()

        # 确保XY轴等比例
        ax.set_aspect('equal', 'box')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Height (m)')
        
        return scatter

    def init_height_map(self, ax, grid_size=0.1, range_radius=5.0):
        """在给定的轴上绘制机器人周围的高度地图"""
        # 获取高度扫描数据
        lidar_data = self.env.unwrapped.scene.sensors["height_scanner_wide"].data.ray_hits_w[0].cpu().numpy()
        lidar_raw = lidar_data.reshape(-1, 3)  # 假设每3个值为一组[x,y,z]坐标
        
        # 提取坐标
        x_points = lidar_raw[:, 0]
        y_points = lidar_raw[:, 1]
        height_values = lidar_raw[:, 2] #- 0.52  # 调整高度值
        height_values = np.full_like(lidar_raw[:, 2], np.nan)
        
        # 获取机器人位置和朝向
        robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
        robot_rot = self.env.unwrapped.scene["robot"].data.root_quat_w[0].cpu().numpy()
        
        # 创建网格
        range_x = (robot_pos[0] - range_radius, robot_pos[0] + range_radius)
        range_y = (robot_pos[1] - range_radius, robot_pos[1] + range_radius)
        
        # 使用散点图绘制高度数据，颜色表示高度
        scatter = ax.scatter(x_points, y_points, c=height_values, cmap='terrain', 
                            s=10, alpha=0.8, vmin=-1.0, vmax=1.0)
        
        # 设置图表属性
        ax.set_xlim(range_x)
        ax.set_ylim(range_y)
        ax.set_title('Terrain Height Map')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()

        # 确保XY轴等比例
        ax.set_aspect('equal', 'box')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Height (m)')
        
        return scatter
        
    def create_animation(self, skill_name="navigate", frames=50, interval=10, range_radius=3,
                         grid_size=0.2, save_path=None):
        """创建动态效果模型可视化"""
        import matplotlib.animation as animation
        
        obs, _ = self.env.get_observations()
        policy = self.env.policies[skill_name]
        
        fig, (ax_height, ax1, ax2) = plt.subplots(1, 3, figsize=(20, 7))
        
        # 成功率和代价的颜色映射
        success_cmap = LinearSegmentedColormap.from_list('success_cmap', ['red', 'yellow', 'green'])
        cost_cmap = LinearSegmentedColormap.from_list('cost_cmap', ['green', 'yellow', 'red'])
        
        # 获取机器人位置
        robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
        robot_rot = self.env.unwrapped.scene["robot"].data.root_quat_w[0].cpu().numpy()
        robot_pos_init = robot_pos.copy()
        range_x = (robot_pos[0] - range_radius, robot_pos[0] + range_radius)
        range_y = (robot_pos[1] - range_radius, robot_pos[1] + range_radius)

        # 计算朝向
        from scipy.spatial.transform import Rotation as R
        r = R.from_quat([robot_rot[1], robot_rot[2], robot_rot[3], robot_rot[0]])
        heading = r.as_euler('xyz')[2]
    
        # 初始化图像 - 先获取一次效果预测以显示初始状态
        success_grid, cost_grid = self.visualize_robot_surroundings(
            grid_size=grid_size, range_radius=range_radius,
            skill_name=skill_name, show_plot=False
        )

        # 初始化图像
        grid_shape = (int((2*range_radius)/grid_size), int((2*range_radius)/grid_size))
        white_grid = np.ones(grid_shape) * np.nan

        # 初始化高度图
        height_scatter = self.init_height_map(ax_height, grid_size, range_radius)

        # 使用全白数组初始化图像
        success_img = ax1.imshow(white_grid, 
                            extent=(range_x[0], range_x[1], range_y[0], range_y[1]),
                            origin='lower', cmap=success_cmap, vmin=0, vmax=1)
        cost_img = ax2.imshow(white_grid,
                            extent=(range_x[0], range_x[1], range_y[0], range_y[1]),
                            origin='lower', cmap=cost_cmap, vmin=0, vmax=1)
        
        # 设置标题和标签
        # 设置高度图的标题和范围
        ax_height.set_title('Terrain Height Map')
        ax_height.set_xlabel('X (m)')
        ax_height.set_ylabel('Y (m)')
        ax_height.legend()

        # 确保XY轴等比例
        ax_height.set_aspect('equal', 'box')

        ax1.set_title(f'{skill_name.capitalize()} Skill - Success Rate')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        fig.colorbar(success_img, ax=ax1, label='Success Probability')

        ax2.set_title(f'{skill_name.capitalize()} Skill - Execution Cost')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        fig.colorbar(cost_img, ax=ax2, label='Normalized Cost')

        # 创建用于显示最高最低值的文本标签
        success_minmax_text = ax1.text(0.25, 0.05, "", 
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax1.transAxes,
            fontsize=10, fontweight='bold')
        
        cost_minmax_text = ax2.text(0.25, 0.05, "", 
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax2.transAxes,
            fontsize=10, fontweight='bold')
       
        # 初始化机器人标记
        robot_plot1, = ax1.plot([], [], 'bo', markersize=10, label='Robot')
        robot_arrow1 = ax1.arrow(0, 0, 0, 0, head_width=0.2, head_length=0.2, fc='blue', ec='blue')
        
        robot_plot2, = ax2.plot([], [], 'bo', markersize=10, label='Robot')
        robot_arrow2 = ax2.arrow(0, 0, 0, 0, head_width=0.2, head_length=0.2, fc='blue', ec='blue')

        robot_plot_height, = ax_height.plot([], [], 'bo', markersize=10, label='Robot')
        robot_arrow_height = ax_height.arrow(0, 0, 0, 0, head_width=0.2, head_length=0.2, fc='blue', ec='blue')


        x_range_min = robot_pos_init[0]-2*range_radius
        x_range_max = robot_pos_init[0]+2*range_radius
        y_range_min = robot_pos_init[1]-2*range_radius
        y_range_max = robot_pos_init[1]+2*range_radius
        ax1.set_xlim(x_range_min, x_range_max)
        ax1.set_ylim(y_range_min, y_range_max)
        ax2.set_xlim(x_range_min, x_range_max)
        ax2.set_ylim(y_range_min, y_range_max)
        ax_height.set_xlim(x_range_min, x_range_max)
        ax_height.set_ylim(y_range_min, y_range_max)
        # ax_height.set_xlim(-20, 20)
        # ax_height.set_ylim(-20, 20)
        
        plt.tight_layout()
        
        def init():
            success_minmax_text.set_text("")
            cost_minmax_text.set_text("")
            return success_img, cost_img, robot_plot1, robot_plot2, robot_plot_height, robot_arrow1, robot_arrow2, robot_arrow_height, success_minmax_text, cost_minmax_text
        
        def animate(i):
            obs, _ = self.env.get_observations()
            
            # 执行行动
            if i > 0:
                with torch.inference_mode():
                    # 执行动作
                    actions = policy(obs)
                    # actions = torch.zeros_like(actions)  # 假设动作是全零的
                    obs, _, _, _ = self.env.step(actions)

                    dist_to_goal = torch.norm(obs[0, 6:8])
                    if dist_to_goal < 0.5:
                        print(f"Task completed successfully in {i} steps.")
                        obs, _ = self.env.reset()

                    # if i%10 == 0:
                    #     obs, _ = self.env.reset()
            
            # 获取机器人位置和朝向
            robot_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[0].cpu().numpy()
            robot_rot = self.env.unwrapped.scene["robot"].data.root_quat_w[0].cpu().numpy()
            robot_state = self.env.unwrapped.scene["robot"].data.root_state_w
            
            # 计算朝向
            from scipy.spatial.transform import Rotation as R
            r = R.from_quat([robot_rot[1], robot_rot[2], robot_rot[3], robot_rot[0]])
            heading = r.as_euler('xyz')[2]
            
            # 创建旋转矩阵
            rotation_matrix = np.array([
                [np.cos(heading), np.sin(heading)],
                [-np.sin(heading), np.cos(heading)]
            ])

            for collection in list(ax_height.collections):
                collection.remove()

            for patch in list(ax_height.patches):
                patch.remove()

            # 更新高度图
            lidar_data = self.env.unwrapped.scene.sensors["height_scanner_wide"].data.ray_hits_w[0].cpu().numpy()
            lidar_raw = lidar_data.reshape(-1, 3)
            # print(lidar_raw[:, 2])
            
            # 直接使用绝对坐标绘制高度图
            animate.height_scatter = ax_height.scatter(
                lidar_raw[:, 0], lidar_raw[:, 1],  # 使用绝对坐标
                c=lidar_raw[:, 2],                  # 高度值
                cmap='terrain', s=10, alpha=0.8, vmin=-1.0, vmax=1.0
            )

            # 初始化高度图
            # height_scatter = self.visualize_height_map(ax_height, grid_size, range_radius)
            
            # 添加机器人朝向标记 - 使用绝对坐标
            r = R.from_quat([robot_rot[1], robot_rot[2], robot_rot[3], robot_rot[0]])
            heading = r.as_euler('xyz')[2]
            dx = np.cos(heading) * 0.5
            dy = np.sin(heading) * 0.5

            # print(self.env.unwrapped.scene["robot"].data.default_root_state)
            # print(self.env.unwrapped.scene["robot"].data)
            # print(self.env.unwrapped.scene["robot"].data.root_pos_w[0])

            # 更新以机器人为中心的范围
            range_x = (robot_pos[0] - range_radius, robot_pos[0] + range_radius)
            range_y = (robot_pos[1] - range_radius, robot_pos[1] + range_radius)
        
            # 更新热图范围
            success_img.set_extent((range_x[0], range_x[1], range_y[0], range_y[1]))
            cost_img.set_extent((range_x[0], range_x[1], range_y[0], range_y[1]))

            # 更新机器人位置标记
            robot_plot1.set_data([robot_pos[0]], [robot_pos[1]])
            robot_plot2.set_data([robot_pos[0]], [robot_pos[1]])
            robot_plot_height.set_data([robot_pos[0]], [robot_pos[1]])
            
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
            if hasattr(animate, 'forward_arrow_height') and animate.forward_arrow_height in ax_height.patches:
                animate.forward_arrow_height.remove()
            
            animate.arrow1 = ax1.arrow(robot_pos[0], robot_pos[1], dx, dy, 
                                       head_width=0.2, head_length=0.2, fc='blue', ec='blue')
            animate.arrow2 = ax2.arrow(robot_pos[0], robot_pos[1], dx, dy, 
                                       head_width=0.2, head_length=0.2, fc='blue', ec='blue')
            animate.forward_arrow_height = ax_height.arrow(
                robot_pos[0], robot_pos[1], dx, dy, 
                head_width=0.2, head_length=0.2, fc='blue', ec='blue'
            )

            # 计算并更新热图 - 修改这部分以解决引用错误
            # if not hasattr(animate, 'last_success_grid'):
            #     # 第一次调用，初始化网格数据
            #     success_grid, cost_grid = self.visualize_robot_surroundings(
            #         grid_size=grid_size, range_radius=range_radius,
            #         skill_name=skill_name, show_plot=False
            #     )
            #     animate.last_success_grid = success_grid
            #     animate.last_cost_grid = cost_grid
            # elif i % 10 == 0:  # 每10帧更新一次热图
            #     success_grid, cost_grid = self.visualize_robot_surroundings(
            #         grid_size=grid_size, range_radius=range_radius,
            #         skill_name=skill_name, show_plot=False
            #     )
            #     animate.last_success_grid = success_grid
            #     animate.last_cost_grid = cost_grid
            # else:
            #     # 使用上次计算的网格数据
            #     success_grid = animate.last_success_grid
            #     cost_grid = animate.last_cost_grid
            
            success_grid, cost_grid = self.visualize_robot_surroundings(
                grid_size=grid_size, range_radius=range_radius,
                skill_name=skill_name, show_plot=False
            )

            # 在create_animation方法中的animate函数内更新热图
            # success_min = np.nanmin(success_grid)
            # success_max = np.nanmax(success_grid)
            # cost_min = np.nanmin(cost_grid)
            # cost_max = np.nanmax(cost_grid)

            # success_vmin = max(0, success_min - 0.1)
            # success_vmax = min(1, success_max + 0.1)
            # cost_vmin = max(0, cost_min - 0.1)
            # cost_vmax = min(1, cost_max + 0.1)

            # success_img.set_clim(success_vmin, success_vmax)
            # cost_img.set_clim(cost_vmin, cost_vmax)
            
            # 计算并显示最高最低值
            success_min = np.nanmin(success_grid)
            success_max = np.nanmax(success_grid)
            cost_min = np.nanmin(cost_grid)
            cost_max = np.nanmax(cost_grid)
            
            # 更新文本显示
            success_minmax_text.set_text(f"Min: {success_min:.4f}, Max: {success_max:.4f}")
            cost_minmax_text.set_text(f"Min: {cost_min:.4f}, Max: {cost_max:.4f}")

            success_img.set_array(success_grid)
            cost_img.set_array(cost_grid)
            
            return (success_img, cost_img, robot_plot1, robot_plot2, robot_plot_height, 
                    animate.arrow1, animate.arrow2, animate.height_scatter, 
                    animate.forward_arrow_height, success_minmax_text, cost_minmax_text)
        
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
        # plt.pause(0.01)
        
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
    data_collector.collect_data_for_skill("navigate", num_trials=50000)
    
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
                # model_trainer.load_model(model_path, input_dim=770)
                model_trainer.load_model(model_path)
                # model_trainer.load_model(model_path, input_dim=963)
                # model_trainer.load_model(model_path, input_dim=191)
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