# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import omni.usd
import numpy as np

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import subtract_frame_transforms, euler_xyz_from_quat

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv

x = 0.3 + 0.6 * torch.arange(10) / 10
y = 0.3 + 0.6 * torch.arange(10) / 10
z = 0.2 + 0.2 * torch.arange(10) / 10

size_tensor = torch.cartesian_prod(
    z.to('cuda:0'),
    y.to('cuda:0'),
    x.to('cuda:0')
)

size_tensor = size_tensor[:, [2, 1, 0]]

def object_yaw(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The yaw of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    _, _, object_yaw = euler_xyz_from_quat(object.data.root_quat_w)
    # print("object_yaw: ", object_yaw)
    return object_yaw.unsqueeze(-1)


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    # print(object_pos_b)
    return object_pos_b

def relative_object_position_in_world_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_rel = object.data.root_pos_w[:, :2] - robot.data.root_state_w[:, :2]
    angle = torch.atan2(object_pos_rel[:, 1], object_pos_rel[:, 0])
    # print("angle: ", angle)
    robot_heading = euler_xyz_from_quat(robot.data.root_state_w[:, 3:7])[2]
    # print("robot_heading: ", robot_heading)
    object_angle_rel = (angle - robot_heading).unsqueeze(1) % (2 * torch.pi) 
    object_angle_rel -= 2 * torch.pi * (object_angle_rel > torch.pi)
    object_pos_rel = torch.cat([object_pos_rel, object_angle_rel], dim=1)
    # object_pos_b, _ = subtract_frame_transforms(
    #     robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    # )
    # print(object_pos_rel)
    return object_pos_rel

def object_size(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    tensor: torch.Tensor = size_tensor
) -> torch.Tensor:
    """The size of the object."""
    object: RigidObject = env.scene[object_cfg.name]
    env_num = env.num_envs
    # object_size = object.data
    # object_size = object.cfg.prim_path
    while len(tensor) < env_num:
        tensor = torch.cat([tensor, tensor], dim=0)
    object_size = tensor[:env_num]
    # print("object_size: ", object_size)
    return object_size

def object_size_specific(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The size of the object."""
    object: RigidObject = env.scene[object_cfg.name]
    object_size = object.cfg.spawn.size
    object_size = torch.tensor(object_size, device='cuda:0').unsqueeze(0)
    # print("object_size: ", object_size)
    return object_size

def robot_pos_w(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The position of the robot in the world frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    robot_pos_w = robot.data.root_pos_w[:, :7]
    # print("robot_pos_w: ", robot_pos_w)
    return robot_pos_w

def object_pos_w(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3] - env.scene.env_origins
    # print("object_pos_w: ", object_pos_w)
    return object_pos_w


def target_object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    target_object_pos_w = subtract_frame_transforms(
        object.data.root_state_w[:, :3], object.data.root_state_w[:, 3:7], command[:, :3]
    )
    # object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], target_object_pos_w
    )
    # print(object_pos_b)
    return object_pos_b


def target_object_pos_w(env: ManagerBasedRLEnv, command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    _, _, yaw = euler_xyz_from_quat(object.data.root_quat_w)
    x_pos = command[:, 0]*torch.cos(yaw) - command[:, 1]*torch.sin(yaw)
    y_pos = command[:, 0]*torch.sin(yaw) + command[:, 1]*torch.cos(yaw)
    rel_pos = torch.stack([x_pos, y_pos], dim=1)
    tar_pos_w = object.data.root_pos_w[:, :2] + rel_pos - env.scene.env_origins[:, :2]
    # print("tar_pos_w: ", tar_pos_w)
    return tar_pos_w

# 全局缓存变量 - 直接按环境ID索引
_object_category_cache = {}

def object_category(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """获取交互物体的类别ID。
    
    将物体分为三类：
    - 类型0: 包含"CardBox"的物体
    - 类型1: 包含"Barel"的物体
    - 类型2: 包含"Chair"的物体
    
    Args:
        env: 环境实例。
        object_cfg: 物体的配置。默认为"object"。
        
    Returns:
        tensor: 物体类别的ID编码 形状为(num_envs, 1)。
    """
    global _object_category_cache
    
    # 准备环境ID和结果tensor
    num_envs = env.num_envs
    device = env.device
    result = torch.zeros((num_envs, 1), dtype=torch.float32, device=device)
    
    # 获取物体实例
    object_view = env.scene[object_cfg.name]
    
    # 缓存键 - 直接使用物体名称
    cache_key = object_cfg.name
    
    # 初始化缓存（如果不存在）
    if cache_key not in _object_category_cache:
        _object_category_cache[cache_key] = {}
    
    # 检查哪些环境需要计算类别
    env_ids_to_process = []
    for env_id in range(num_envs):
        if env_id in _object_category_cache[cache_key]:
            # 已有缓存，直接使用
            result[env_id, 0] = _object_category_cache[cache_key][env_id]
        else:
            # 没有缓存，需要处理
            env_ids_to_process.append(env_id)
    
    # 如果所有环境都有缓存，直接返回
    if not env_ids_to_process:
        return result
    
    # 获取stage
    stage = omni.usd.get_context().get_stage()
    
    # 为未缓存的环境计算类别
    for env_id in env_ids_to_process:
        try:
            # 获取物体的prim路径
            prim_path = object_view.root_physx_view.prim_paths[env_id]
            prim = stage.GetPrimAtPath(prim_path)
            
            category_id = 0  # 默认类别
            
            if prim.IsValid():
                # 获取子Prim（通常第一个子Prim是mesh）
                children = list(prim.GetChildren())
                
                # 跳过"Looks"和其他非网格prim
                mesh_children = [child for child in children if "Looks" not in str(child.GetPath())]
                
                if mesh_children:
                    # 取第一个mesh子prim的名称
                    child_name = mesh_children[0].GetName().lower()
                    
                    # 简单规则匹配
                    if "cardbox" in child_name:
                        category_id = 0  # 纸箱类型
                    elif "barel" in child_name:
                        category_id = 1  # 桶类型
                    elif "chair" in child_name:
                        category_id = 2  # 
                    else:
                        category_id = 1  # 默认桶类型
            
            # 更新结果和缓存
            result[env_id, 0] = float(category_id)
            _object_category_cache[cache_key][env_id] = category_id
            
        except Exception as e:
            print(f"处理环境{env_id}时出错: {e}")
            # 出错时使用默认值0
            result[env_id, 0] = 0.0
            _object_category_cache[cache_key][env_id] = 0
    
    # print("object_category: ", result)
    return result

# 点云缓存
_point_cloud_cache = {}

def object_shape_pc(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("interactive_object"),
    num_points: int = 64,
    min_height: float = 0.0,
    max_height: float = 0.5
) -> torch.Tensor:
    """获取物体在特定高度区间内的形状点云，并展平为一维向量。
    
    通过对网格表面采样，获取更完整的点云表示。
    
    Args:
        env: 环境实例。
        object_cfg: 物体的配置。
        num_points: 采样的点云数量。
        min_height: 过滤的最小高度 相对于物体中心 默认为0.0
        max_height: 过滤的最大高度 相对于物体中心 默认为0.5
        
    Returns:
        tensor: 形状为 (num_envs, num_points*3) 的展平点云数据。
    """
    global _point_cloud_cache
    
    # 获取物体实例
    object_view = env.scene[object_cfg.name]
    
    # 准备结果tensor - 注意这里改为展平形状
    num_envs = env.num_envs
    device = env.device
    # 先创建3D点云
    point_cloud_3d = torch.zeros((num_envs, num_points, 3), dtype=torch.float32, device=device)
    
    # 缓存键
    cache_key = f"{object_cfg.name}_{num_points}_{min_height}_{max_height}"
    
    # 初始化缓存（如果不存在）
    if cache_key not in _point_cloud_cache:
        _point_cloud_cache[cache_key] = {}
    
    # 检查哪些环境需要计算点云
    env_ids_to_process = []
    for env_id in range(num_envs):
        # 创建环境特定的缓存键
        try:
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            prim_path = object_view.root_physx_view.prim_paths[env_id]
            prim = stage.GetPrimAtPath(prim_path)
            
            if prim.IsValid():
                children = list(prim.GetChildren())
                mesh_children = [child for child in children if "Looks" not in str(child.GetPath())]
                if mesh_children:
                    # 用第一个mesh子prim的名称作为缓存键的一部分
                    object_type = mesh_children[0].GetName().lower()
                    cache_id = f"{env_id}_{object_type}"
                    
                    if cache_id in _point_cloud_cache[cache_key]:
                        # 已有缓存，直接使用
                        point_cloud_3d[env_id] = _point_cloud_cache[cache_key][cache_id]
                    else:
                        # 没有缓存，需要处理
                        env_ids_to_process.append((env_id, object_type, prim_path))
                else:
                    env_ids_to_process.append((env_id, "unknown", prim_path))
            else:
                env_ids_to_process.append((env_id, "invalid", prim_path))
        except Exception as e:
            print(f"处理环境{env_id}的缓存键时出错: {e}")
            env_ids_to_process.append((env_id, "error", ""))
    
    # 如果所有环境都有缓存，直接返回展平的结果
    if not env_ids_to_process:
        return point_cloud_3d.reshape(num_envs, num_points * 3)
    
    # 处理需要计算的环境
    import omni.usd
    from pxr import Usd, UsdGeom, Gf
    import numpy as np
    
    # 获取stage
    stage = omni.usd.get_context().get_stage()
    
    for env_id, object_type, prim_path in env_ids_to_process:
        try:
            # 固定随机种子，确保相同物体类型生成相同的点云
            seed = int(stable_hash(object_type) % 10000000)
            np_state = np.random.get_state()
            np.random.seed(seed)  # 使用物体类型作为随机种子
            
            prim = stage.GetPrimAtPath(prim_path)
            
            # 收集所有网格的点和面
            all_points = []
            all_faces = []
            all_face_counts = []
            point_offset = 0
            
            # 遍历物体的层级结构
            def process_prim_recursively(current_prim, parent_transform=None):
                nonlocal point_offset
                
                # 跳过特定类型或名称的prim
                if "Looks" in str(current_prim.GetPath()):
                    return
                
                # 获取当前Prim的局部变换
                xformable = UsdGeom.Xformable(current_prim)
                local_transform = xformable.GetLocalTransformation()
                local_matrix = np.array(local_transform).reshape(4, 4)
                
                # 计算当前Prim相对于根节点的累积变换
                if parent_transform is not None:
                    current_transform = np.dot(parent_transform, local_matrix)
                else:
                    current_transform = local_matrix
                
                # 检查是否为Mesh类型
                if current_prim.GetTypeName() == "Mesh":
                    mesh = UsdGeom.Mesh(current_prim)
                    
                    # 获取顶点
                    vertices = mesh.GetPointsAttr().Get()
                    if vertices is not None and len(vertices) > 0:
                        # 转换为numpy数组
                        vertices_np = np.array([(v[0], v[1], v[2]) for v in vertices])
                        
                        # 获取面信息
                        face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
                        face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
                        
                        if face_vertex_counts is not None and face_vertex_indices is not None:
                            # 转换顶点到世界坐标系
                            homogeneous = np.ones((len(vertices_np), 4))
                            homogeneous[:, :3] = vertices_np
                            transformed_vertices = np.dot(homogeneous, current_transform.T)[:, :3]
                            
                            all_points.append(transformed_vertices)
                            
                            # 记录面信息，用于后续采样
                            face_start = 0
                            for face_verts in face_vertex_counts:
                                # 提取当前面的顶点索引
                                face_indices = face_vertex_indices[face_start:face_start + face_verts]
                                face_indices = [idx + point_offset for idx in face_indices]
                                all_faces.append(face_indices)
                                all_face_counts.append(face_verts)
                                face_start += face_verts
                            
                            # 更新顶点偏移量
                            point_offset += len(vertices_np)
                
                # 递归处理子Prim，传递当前累积变换
                for child in current_prim.GetChildren():
                    process_prim_recursively(child, current_transform)
            
            # 开始递归处理
            if prim.IsValid():
                process_prim_recursively(prim)
            
            # 合并所有点和处理面信息
            if all_points and len(all_points) > 0:
                # 合并所有顶点
                all_vertices = np.vstack(all_points)
                
                # 使用三角形面来生成更密集的点云
                surface_points = []
                
                # 遍历所有面，对每个面进行采样
                face_start_idx = 0
                for face_idx, face_verts in enumerate(all_face_counts):
                    # 获取当前面的顶点索引
                    face_indices = all_faces[face_idx]
                    
                    # 对三角形面进行处理
                    if face_verts == 3:  # 三角形
                        triangle_vertices = all_vertices[face_indices]
                        # 对三角形表面进行均匀采样
                        surface_points.extend(sample_triangle(triangle_vertices, 10))  # 每个三角形采样10个点
                    elif face_verts == 4:  # 四边形 - 拆分为两个三角形
                        quad_vertices = all_vertices[face_indices]
                        # 将四边形拆分为两个三角形
                        triangle1 = quad_vertices[[0, 1, 2]]
                        triangle2 = quad_vertices[[0, 2, 3]]
                        # 采样两个三角形
                        surface_points.extend(sample_triangle(triangle1, 5))
                        surface_points.extend(sample_triangle(triangle2, 5))
                    elif face_verts > 4:  # 多边形 - 使用简单的扇形三角剖分
                        for i in range(1, face_verts - 1):
                            tri_vertices = all_vertices[[face_indices[0], face_indices[i], face_indices[i+1]]]
                            surface_points.extend(sample_triangle(tri_vertices, 3))
                
                # 添加原始顶点，确保关键点被保留
                surface_points.extend(all_vertices)
                
                # 转换为numpy数组
                if surface_points:
                    sampled_points = np.array(surface_points)
                    
                    # 过滤高度范围
                    height_mask = (sampled_points[:, 2] >= min_height) & (sampled_points[:, 2] <= max_height)
                    filtered_points = sampled_points[height_mask]
                    
                    # 如果过滤后点太少，使用全部点
                    if len(filtered_points) < num_points * 0.5 and len(sampled_points) >= num_points:
                        filtered_points = sampled_points
                    
                    # 确保有足够的点
                    if len(filtered_points) >= num_points:
                        # 随机采样所需数量的点
                        indices = np.random.choice(len(filtered_points), num_points, replace=False)
                        final_points = filtered_points[indices]
                    elif len(filtered_points) > 0:
                        # 重复采样以达到所需数量
                        indices = np.random.choice(len(filtered_points), num_points, replace=True)
                        final_points = filtered_points[indices]
                    else:
                        # 如果过滤后没有点，使用原始点
                        if len(sampled_points) >= num_points:
                            indices = np.random.choice(len(sampled_points), num_points, replace=False)
                            final_points = sampled_points[indices]
                        elif len(sampled_points) > 0:
                            indices = np.random.choice(len(sampled_points), num_points, replace=True)
                            final_points = sampled_points[indices]
                        else:
                            # 如果没有点，创建空点云
                            final_points = np.zeros((num_points, 3))
                    
                    # 保存到结果和缓存
                    points_tensor = torch.tensor(final_points, dtype=torch.float32, device=device)
                    point_cloud_3d[env_id] = points_tensor
                    
                    # 存入缓存
                    cache_id = f"{env_id}_{object_type}"
                    _point_cloud_cache[cache_key][cache_id] = points_tensor.clone()
                else:
                    # 没有成功生成点，使用替代方案
                    substitute_points = create_substitute_point_cloud(num_points, min_height, max_height, device, seed=seed)
                    point_cloud_3d[env_id] = substitute_points
                    
                    # 存入缓存
                    cache_id = f"{env_id}_{object_type}"
                    _point_cloud_cache[cache_key][cache_id] = substitute_points.clone()
            else:
                # 没有找到有效的网格，使用替代方案
                substitute_points = create_substitute_point_cloud(num_points, min_height, max_height, device, seed=seed)
                point_cloud_3d[env_id] = substitute_points
                
                # 存入缓存
                cache_id = f"{env_id}_{object_type}"
                _point_cloud_cache[cache_key][cache_id] = substitute_points.clone()
            
            # 恢复随机状态
            np.random.set_state(np_state)
                
        except Exception as e:
            print(f"处理环境{env_id}的点云时出错: {e}")
            import traceback
            traceback.print_exc()
            
            # 出错时使用替代方案
            seed = int(stable_hash(f"error_{object_type}") % 10000000)
            substitute_points = create_substitute_point_cloud(num_points, min_height, max_height, device, seed=seed)
            point_cloud_3d[env_id] = substitute_points
    
    # 将3D点云展平为1D向量并返回
    return point_cloud_3d.reshape(num_envs, num_points * 3)

def sample_triangle(vertices, num_samples):
    """对三角形表面进行均匀采样"""
    if len(vertices) != 3:
        return []
    
    samples = []
    for _ in range(num_samples):
        # 生成重心坐标
        r1 = np.random.random()
        r2 = np.random.random()
        
        # 确保是均匀分布
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
            
        # 计算第三个重心坐标
        r3 = 1 - r1 - r2
        
        # 使用重心坐标计算点
        point = vertices[0] * r1 + vertices[1] * r2 + vertices[2] * r3
        samples.append(point)
    
    return samples

def create_substitute_point_cloud(num_points, min_height, max_height, device):
    """当无法获取真实点云时，创建一个简单的替代点云。"""
    import numpy as np
    
    # 创建一个简单的盒状点云
    x = np.random.uniform(-0.5, 0.5, num_points)
    y = np.random.uniform(-0.5, 0.5, num_points)
    z = np.random.uniform(min_height, max_height, num_points)
    
    points = np.column_stack([x, y, z])
    return torch.tensor(points, dtype=torch.float32, device=device)


#######################################################
# 全局缓存变量 - 按物体类型缓存
_object_type_point_clouds = {}

# 全局缓存变量 - 环境到物体类型的映射
_env_to_object_type = {}

# 第一次访问时初始化的标志
_point_cloud_initialized = False

def calculate_object_type_mapping(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("interactive_object")
) -> dict:
    """计算每个环境实例对应的物体类型，只需计算一次。
    
    Args:
        env: 环境实例
        object_cfg: 物体配置
        
    Returns:
        dict: 环境ID到物体类型的映射
    """
    global _env_to_object_type
    
    # 缓存键
    cache_key = f"{env.cfg.__class__.__name__}_{object_cfg.name}"
    
    # 如果已经缓存，直接返回
    if cache_key in _env_to_object_type:
        return _env_to_object_type[cache_key]
    
    # 初始化映射
    _env_to_object_type[cache_key] = {}
    
    # 获取物体实例
    object_view = env.scene[object_cfg.name]
    
    # 获取stage
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    
    # 为每个环境获取物体类型
    for env_id in range(env.num_envs):
        try:
            # 获取物体的prim路径
            prim_path = object_view.root_physx_view.prim_paths[env_id]
            prim = stage.GetPrimAtPath(prim_path)
            
            object_type = "unknown"
            if prim.IsValid():
                children = list(prim.GetChildren())
                mesh_children = [child for child in children if "Looks" not in str(child.GetPath())]
                if mesh_children:
                    object_type = mesh_children[0].GetName().lower()
            
            # 存储映射
            _env_to_object_type[cache_key][env_id] = object_type
            
        except Exception as e:
            print(f"获取环境{env_id}的物体类型时出错: {e}")
            _env_to_object_type[cache_key][env_id] = "unknown"
    
    # 返回映射
    return _env_to_object_type[cache_key]

def stable_hash(text):
    value = 0
    for char in text:
        value = (value * 31 + ord(char)) & 0xFFFFFFFF
    return value

def generate_point_cloud_for_type(
    env: ManagerBasedRLEnv, 
    object_type: str,
    prim_path: str,
    num_points: int = 64,
    min_height: float = -0.5,
    max_height: float = 0.5,
    device: torch.device = None
) -> torch.Tensor:
    """为特定类型的物体生成点云。
    
    Args:
        env: 环境实例
        object_type: 物体类型名称
        prim_path: 物体的prim路径
        num_points: 点数
        min_height: 最小高度
        max_height: 最大高度
        device: 计算设备
        
    Returns:
        torch.Tensor: 生成的点云
    """
    import numpy as np

    seed = int(stable_hash(object_type) % 10000000)
    np_state = np.random.get_state()
    np.random.seed(seed)  # 使用物体类型作为随机种子

    if device is None:
        device = env.device
    
    import omni.usd
    from pxr import Usd, UsdGeom, Gf
    import numpy as np
    
    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    
    # 收集点和面
    all_points = []
    all_faces = []
    all_face_counts = []
    point_offset = 0
    
    # 递归处理prim
    def process_prim_recursively(current_prim, parent_transform=None):
        nonlocal point_offset
        
        # 跳过Looks
        if "Looks" in str(current_prim.GetPath()):
            return
        
        # 处理变换
        xformable = UsdGeom.Xformable(current_prim)
        local_transform = xformable.GetLocalTransformation()
        local_matrix = np.array(local_transform).reshape(4, 4)
        
        # 计算累积变换
        if parent_transform is not None:
            current_transform = np.dot(parent_transform, local_matrix)
        else:
            current_transform = local_matrix
        
        # 处理Mesh
        if current_prim.GetTypeName() == "Mesh":
            mesh = UsdGeom.Mesh(current_prim)
            
            vertices = mesh.GetPointsAttr().Get()
            if vertices is not None and len(vertices) > 0:
                vertices_np = np.array([(v[0], v[1], v[2]) for v in vertices])
                
                face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
                face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
                
                if face_vertex_counts is not None and face_vertex_indices is not None:
                    # 变换顶点
                    homogeneous = np.ones((len(vertices_np), 4))
                    homogeneous[:, :3] = vertices_np
                    transformed_vertices = np.dot(homogeneous, current_transform.T)[:, :3]
                    
                    all_points.append(transformed_vertices)
                    
                    # 处理面
                    face_start = 0
                    for face_verts in face_vertex_counts:
                        face_indices = face_vertex_indices[face_start:face_start + face_verts]
                        face_indices = [idx + point_offset for idx in face_indices]
                        all_faces.append(face_indices)
                        all_face_counts.append(face_verts)
                        face_start += face_verts
                    
                    point_offset += len(vertices_np)
        
        # 递归处理子节点
        for child in current_prim.GetChildren():
            process_prim_recursively(child, current_transform)
    
    # 开始处理
    process_prim_recursively(prim)
    
    # 检查是否有有效数据
    if not all_points:
        # 没有找到有效的网格，使用替代方案
        return create_substitute_point_cloud(num_points, min_height, max_height, device)
    
    # 合并顶点
    all_vertices = np.vstack(all_points)
    
    # 采样面
    surface_points = []
    
    # 处理每个面
    for face_idx, face_verts in enumerate(all_face_counts):
        face_indices = all_faces[face_idx]
        
        if face_verts == 3:  # 三角形
            triangle_vertices = all_vertices[face_indices]
            # 采样点
            surface_points.extend(sample_triangle(triangle_vertices, 10))
        elif face_verts == 4:  # 四边形
            quad_vertices = all_vertices[face_indices]
            # 分解为两个三角形
            triangle1 = quad_vertices[[0, 1, 2]]
            triangle2 = quad_vertices[[0, 2, 3]]
            surface_points.extend(sample_triangle(triangle1, 5))
            surface_points.extend(sample_triangle(triangle2, 5))
        elif face_verts > 4:  # 多边形
            # 简单的扇形三角剖分
            for i in range(1, face_verts - 1):
                tri_vertices = all_vertices[[face_indices[0], face_indices[i], face_indices[i+1]]]
                surface_points.extend(sample_triangle(tri_vertices, 5))
    
    # 添加原始顶点
    surface_points.extend(all_vertices)
    
    # 检查是否成功生成点
    if not surface_points:
        return create_substitute_point_cloud(num_points, min_height, max_height, device)
    
    # 转换为numpy数组
    sampled_points = np.array(surface_points)
    
    # 过滤高度
    height_mask = (sampled_points[:, 2] >= min_height) & (sampled_points[:, 2] <= max_height)
    filtered_points = sampled_points[height_mask]
    
    # 如果过滤后点太少，使用全部点
    if len(filtered_points) < num_points * 0.5 and len(sampled_points) >= num_points:
        filtered_points = sampled_points
    
    # 采样最终点云
    if len(filtered_points) >= num_points:
        indices = np.random.choice(len(filtered_points), num_points, replace=False)
        final_points = filtered_points[indices]
    elif len(filtered_points) > 0:
        indices = np.random.choice(len(filtered_points), num_points, replace=True)
        final_points = filtered_points[indices]
    else:
        return create_substitute_point_cloud(num_points, min_height, max_height, device)
    
    # 转换为tensor
    points_tensor = torch.tensor(final_points, dtype=torch.float32, device=device)
    
    np.random.set_state(np_state)
    return points_tensor

def generate_all_point_clouds(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("interactive_object"),
    num_points: int = 64,
    min_height: float = -0.5,
    max_height: float = 0.5
) -> None:
    """为所有类型的物体生成点云，并存储在全局缓存中。
    
    Args:
        env: 环境实例
        object_cfg: 物体配置
        num_points: 点数量
        min_height: 最小高度
        max_height: 最大高度
    """
    global _object_type_point_clouds, _point_cloud_initialized
    
    # 如果已经初始化过，跳过
    if _point_cloud_initialized:
        return
    
    # 设置标志
    _point_cloud_initialized = True
    
    # 缓存键
    cache_key = f"{object_cfg.name}_{num_points}_{min_height}_{max_height}"
    
    # 初始化缓存
    if cache_key not in _object_type_point_clouds:
        _object_type_point_clouds[cache_key] = {}
    
    # 获取环境到物体类型的映射
    env_to_type = calculate_object_type_mapping(env, object_cfg)
    
    # 获取物体实例
    object_view = env.scene[object_cfg.name]
    
    # 收集所有唯一的物体类型
    unique_types = set(env_to_type.values())
    
    print(f"开始为{len(unique_types)}种物体类型生成点云...")
    import time
    start_time = time.time()
    
    # 为每种类型生成点云
    for object_type in unique_types:
        # 查找该类型的第一个实例
        for env_id, type_name in env_to_type.items():
            if type_name == object_type:
                # 获取物体的prim路径
                prim_path = object_view.root_physx_view.prim_paths[env_id]
                
                # 生成点云
                print(f"生成类型 '{object_type}' 的点云...")
                try:
                    point_cloud = generate_point_cloud_for_type(
                        env, object_type, prim_path, num_points, min_height, max_height
                    )
                    
                    # 存储到缓存
                    _object_type_point_clouds[cache_key][object_type] = point_cloud
                    print(f"  成功生成 '{object_type}' 的点云，包含 {num_points} 个点")
                    
                except Exception as e:
                    print(f"  为类型 '{object_type}' 生成点云时出错: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # 使用默认点云
                    _object_type_point_clouds[cache_key][object_type] = create_substitute_point_cloud(
                        num_points, min_height, max_height, env.device
                    )
                
                # 每种类型只处理一次
                break
    
    # 添加默认点云
    _object_type_point_clouds[cache_key]["default"] = create_substitute_point_cloud(
        num_points, min_height, max_height, env.device
    )
    
    elapsed_time = time.time() - start_time
    print(f"点云生成完成，耗时: {elapsed_time:.2f}秒")

def object_shape_pc_optimized(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("interactive_object"),
    num_points: int = 64,
    min_height: float = -0.5,
    max_height: float = 0.5
) -> torch.Tensor:
    """获取物体点云的优化版本。
    
    通过预计算和类型映射大幅提高性能。
    
    Args:
        env: 环境实例
        object_cfg: 物体配置
        num_points: 点数量
        min_height: 最小高度
        max_height: 最大高度
        
    Returns:
        torch.Tensor: 点云数据，形状为(num_envs, num_points*3)
    """
    global _object_type_point_clouds, _env_to_object_type, _point_cloud_initialized
    
    # 初始化所有点云（如果尚未完成）
    if not _point_cloud_initialized:
        generate_all_point_clouds(env, object_cfg, num_points, min_height, max_height)
    
    # 缓存键
    cache_key = f"{object_cfg.name}_{num_points}_{min_height}_{max_height}"
    env_cache_key = f"{env.cfg.__class__.__name__}_{object_cfg.name}"
    
    # 准备输出tensor
    num_envs = env.num_envs
    device = env.device
    point_cloud_3d = torch.zeros((num_envs, num_points, 3), dtype=torch.float32, device=device)
    
    # 按类型批量分配点云
    env_ids_by_type = {}
    env_to_type = _env_to_object_type.get(env_cache_key, {})
    
    # 检查是否已计算环境到类型的映射
    if not env_to_type:
        env_to_type = calculate_object_type_mapping(env, object_cfg)
    
    # 按类型分组环境ID
    for env_id, object_type in env_to_type.items():
        if env_id >= num_envs:
            continue
            
        if object_type not in env_ids_by_type:
            env_ids_by_type[object_type] = []
        env_ids_by_type[object_type].append(env_id)
    
    # 批量分配点云
    for object_type, env_ids in env_ids_by_type.items():
        # 转换为torch.LongTensor以便索引
        env_indices = torch.tensor(env_ids, dtype=torch.long, device=device)
        
        # 获取点云
        if object_type in _object_type_point_clouds[cache_key]:
            point_cloud = _object_type_point_clouds[cache_key][object_type]
        else:
            point_cloud = _object_type_point_clouds[cache_key]["default"]
        
        # 一次性分配给所有相同类型的环境
        point_cloud_3d.index_copy_(0, env_indices, point_cloud.unsqueeze(0).expand(len(env_indices), -1, -1))
    
    # 展平并返回
    return point_cloud_3d.reshape(num_envs, num_points * 3)