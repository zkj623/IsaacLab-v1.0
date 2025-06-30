# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
from dataclasses import MISSING

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from omni.isaac.lab.envs import ManagerBasedRLEnvCfg
from omni.isaac.lab.managers import CurriculumTermCfg as CurrTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import RewardTermCfg as RewTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import CameraCfg, RayCasterCameraCfg, ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
# from omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.go2.climb_env_cfg import UnitreeGo2ClimbEnvCfg
# import omni.isaac.lab_tasks.manager_based.navigation.mdp as mdp

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from omni.isaac.lab.terrains.config.navigate_anywhere import NAVIGATE_ANYWHERE_CFG, PIT_CROSS_CFG, SIMULATION1_CFG 

LOW_LEVEL_ENV_CFG = UnitreeGo2RoughEnvCfg()
# LOW_LEVEL_Climb_ENV_CFG = UnitreeGo2ClimbEnvCfg()

base_command = {}
def constant_commands(env: ManagerBasedRLEnvCfg) -> torch.Tensor:
    global base_command
    """The generated command from the command generator."""
    tensor_lst = torch.tensor([0, 0, 0], device=env.device).repeat(env.num_envs, 1)
    for i in range(env.num_envs):
        tensor_lst[i] = torch.tensor(base_command[str(i)], device=env.device)
    return tensor_lst

##
# Scene definition
##

sampling_time = 8.0

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        # terrain_type="generator",
        terrain_type="plane",
        # terrain_generator=ROUGH_TERRAINS_CFG,
        # terrain_generator=NAVIGATE_ANYWHERE_CFG,
        terrain_generator=SIMULATION1_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    office = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Office",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0], rot=[0, 0, 0, 1.0]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Office/office.usd"),
    )

    # carla = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Carla",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, 0], rot=[0, 0, 0, 1.0]),
    #     spawn=UsdFileCfg(usd_path="/home/zhou/IsaacLab/carla_export/new_carla_export/carla.usd"),
    # )
    
    # robots
    robot: ArticulationCfg = MISSING
    # target object
    box_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box_1",
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[-10, 0, 0], rot=[1, 0, 0, 0]),
        # simulation1
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[-1.2, -3.1, 0.2], rot=[1, 0, 0, 0]),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.6, -1.55, 0.2], rot=[1, 0, 0, 0]),
        # simulation2
        init_state=RigidObjectCfg.InitialStateCfg(pos=[2.7, 1.0, 0.2], rot=[1, 0, 0, 0]),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.5, -1.2, 0.2], rot=[0.9997,0,0,-0.0262]),
        # simulation3
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[1.8, 1.2, 0.3], rot=[1, 0, 0, 0]),
        # simulation4
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[-1.0, 0.4, 0.3], rot=[1, 0, 0, 0]),
        # simulation5
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.5, 1.0, 0.3], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            # simulation1
            # size=(0.8, 0.8, 0.35),
            # simulation2
            size=(0.9, 0.9, 0.22),
            # simulation3
            # size=(0.9, 0.9, 0.35),
            # simulation4
            # size=(0.7, 0.9, 0.3), 
            # simulation4
            # size=(0.4, 0.5, 0.4),
            # rigid_props=RigidBodyPropertiesCfg(
            #     solver_position_iteration_count=16,
            #     solver_velocity_iteration_count=1,
            #     max_angular_velocity=1000.0,
            #     max_linear_velocity=1000.0,
            #     max_depenetration_velocity=5.0,
            #     disable_gravity=False,
            # ),
            rigid_props=RigidBodyPropertiesCfg(),
            # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            # simulation 1 & 2
            mass_props=sim_utils.MassPropertiesCfg(density=10.0),
            # simulation 3
            # mass_props=sim_utils.MassPropertiesCfg(density=40.0),
            # simulation 3
            # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.8)),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wall_Board/Cardboard.mdl",
                # mdl_path="/home/zkj/IsaacLab/Cardboard.mdl",
                # mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wall_Board/MDF.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
        ),
        debug_vis=False,
    )
    box_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box_2",
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[-10, 0, 0], rot=[1, 0, 0, 0]),
        # simulation1
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[1.9, 1.8, 0.25], rot=[1, 0, 0, 0]),
        # simulation2
        init_state=RigidObjectCfg.InitialStateCfg(pos=[0.9, 2.8, 0.2], rot=[1, 0, 0, 0]),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[-1.8, -3.3, 0.2], rot=[0.9659, 0, 0, -0.2588]),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[-2.8, -2.38, 0.2], rot=[0.9997,0,0,0.0262]),
        # simulation3
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[2.5, 1.5, 0.3], rot=[1, 0, 0, 0]),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[2.5, 0.3, 0.3], rot=[1, 0, 0, 0]),
        # simulation4
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[2.5, 1.0, 0.3], rot=[1, 0, 0, 0]),
        # simulation5
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[2.5, 1.0, 0.3], rot=[1, 0, 0, 0]),
        spawn=sim_utils.CuboidCfg(
            # simulation1
            # size=(0.9, 0.4, 0.4),
            # simulation2
            size=(0.6, 0.7, 0.18),
            # simulation3
            # size=(0.9, 0.6, 0.35),
            # size=(0.9, 0.6, 0.18),
            # simulation4
            # size=(1.0, 0.8, 0.35),
            # simulation5
            # size=(0.8, 0.6, 0.3),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            # mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            mass_props=sim_utils.MassPropertiesCfg(density=10.0), # 50.0
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.4, 0.4)),
            visual_material=sim_utils.MdlFileCfg(
                # mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/vMaterials_2/Paper/Cardboard.mdl",
                mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/vMaterials_2/Paper/Cardboard_Low_Quality.mdl",
                project_uvw=True,
                texture_scale=(0.5, 0.5),
            ),
        ),
        debug_vis=False,
    )
    box_3: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Box_3",
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[-10, 0, 0], rot=[1, 0, 0, 0]),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[-10, 0, 0], rot=[1, 0, 0, 0]),
        # simulation1
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, 3.0, 0.2], rot=[1, 0, 0, 0]),
        # simulation2
        init_state=RigidObjectCfg.InitialStateCfg(pos=[2.5, -2.8, 0.3], rot=[0.9659, 0, 0, -0.2588]),#0.9997,0,0,-0.05
        # simulation3
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.5, 2.0, 0.3], rot=[0.9997,0,0,-0.05]),#0.9997,0,0,-0.05
        # simulation4
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[-3.2, -2.0, 0.3], rot=[0.9997,0,0,-0.05]),#0.9997,0,0,-0.05
        # simulation5
        # init_state=RigidObjectCfg.InitialStateCfg(pos=[-3.2, -2.0, 0.3], rot=[0.9997,0,0,-0.05]),#0.9997,0,0,-0.05
        spawn=sim_utils.CuboidCfg(
            size=(0.4, 0.6, 0.30),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            # mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            # mass_props=sim_utils.MassPropertiesCfg(density=10.0),
            mass_props=sim_utils.MassPropertiesCfg(density=20.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.8, 0.4)),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/vMaterials_2/Paper/Cardboard.mdl",
                # mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/vMaterials_2/Paper/Cardboard_Low_Quality.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
        ),
        debug_vis=False,
    )
    # box_4: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Box_4",
    #     # init_state=RigidObjectCfg.InitialStateCfg(pos=[-10, 0, 0], rot=[1, 0, 0, 0]),
    #     # simulation4
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.7, -0.5, 0.3], rot=[0.9997,0,0,-0.05]),#0.9997,0,0,-0.05
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.6, 0.6, 0.30),
    #         rigid_props=RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=1,
    #             max_angular_velocity=1000.0,
    #             max_linear_velocity=1000.0,
    #             max_depenetration_velocity=5.0,
    #             disable_gravity=False,
    #         ),
    #         # mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
    #         mass_props=sim_utils.MassPropertiesCfg(density=10.0),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.8, 0.4)),
    #         visual_material=sim_utils.MdlFileCfg(
    #             mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/vMaterials_2/Paper/Cardboard.mdl",
    #             # mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/vMaterials_2/Paper/Cardboard_Low_Quality.mdl",
    #             project_uvw=True,
    #             texture_scale=(0.25, 0.25),
    #         ),
    #     ),
    #     debug_vis=False,
    # )
    # platform: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Platform",
    #     # simulation1
    #     # init_state=RigidObjectCfg.InitialStateCfg(pos=[-1.2, -3.1, 0.2], rot=[1, 0, 0, 0]),
    #     # simulation2
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.5, -0.5, 0.2], rot=[1, 0, 0, 0]),
    #     # init_state=RigidObjectCfg.InitialStateCfg(pos=[-0.5, -1.2, 0.2], rot=[0.9997,0,0,-0.0262]),
    #     spawn=sim_utils.CuboidCfg(
    #         # simulation1
    #         # size=(0.8, 0.8, 0.35),
    #         # simulation2
    #         size=(0.8, 2.0, 0.50),
    #         # size=(0.3, 0.3, 0.2),
    #         # rigid_props=RigidBodyPropertiesCfg(
    #         #     solver_position_iteration_count=16,
    #         #     solver_velocity_iteration_count=1,
    #         #     max_angular_velocity=1000.0,
    #         #     max_linear_velocity=1000.0,
    #         #     max_depenetration_velocity=5.0,
    #         #     disable_gravity=False,
    #         # ),
    #         rigid_props=RigidBodyPropertiesCfg(),
    #         # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.MdlFileCfg(
    #             mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
    #             project_uvw=True,
    #             texture_scale=(0.25, 0.25),
    #         ),
    #     ),
    #     debug_vis=False,
    # )

    # sensors
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane", "distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=9.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.410, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    robot_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/third_view_cam",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=9.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 1.0e5)
        ),
        # offset=CameraCfg.OffsetCfg(pos=(0.0, -0.5, 0.7), rot=(0.683, -0.183, 0.183, 0.683), convention="world"), 
        offset=CameraCfg.OffsetCfg(pos=(-0.8, -0.8, 0.8), rot=(0.9, -0.105, 0.271, 0.325), convention="world"),  # (0.854, -0.354, 0.146, 0.354) (0.854, 0.146, 0.354, 0.354)
        # offset=CameraCfg.OffsetCfg(pos=(-1.0, 0.0, 1.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    height_scanner_env = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[6.0, 3.5]),
        debug_vis=True,
        mesh_prim_paths=["/World/envs/env_0/Office"],
        # mesh_prim_paths=["/World/ground"],
    )

    # object_height_scanner = RayCasterCfg(
    #     # prim_path="{ENV_REGEX_NS}/Robot/base",
    #     prim_path="/World/envs/env_0/Box_1",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=True,
    #     mesh_prim_paths=["/World/envs/env_0/Box_1"],
    #     # mesh_prim_paths=["/World/ground"],
    #     # mesh_prim_paths=["/World/envs/env_0/Box_1", "/World/envs/env_0/Box_2", "/World/envs/env_0/Box_3"],
    # )

    height_scanner2 = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(2.0, 0.0, 20.0)),
        attach_yaw_only=True,
        # pattern_cfg=patterns.LidarPatternCfg(
        #     channels = 200,
        #     vertical_fov_range=[-90, 90],
        #     horizontal_fov_range = [-90,90],
        #     horizontal_res=1),
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[6.0, 3.5]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    
    # distant_light = AssetBaseCfg(
    #     prim_path="/World/DistantLight",
    #     spawn=sim_utils.DistantLightCfg(intensity=2500.0),
    #     init_state=AssetBaseCfg.InitialStateCfg(rot=(0.738, 0.477, 0.477, 0.0)),
    # )




##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # pose_command = mdp.TerrainBasedPose2dCommandCfg(
    #     asset_name="robot",
    #     simple_heading=False,
    #     resampling_time_range=(15.0, 15.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(1.4, 1.5), pos_y=(0.0, 0.1), heading=(math.pi, math.pi)),
    # )

    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=False,
        resampling_time_range=(sampling_time, sampling_time),
        debug_vis=False,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(1.4, 1.5), pos_y=(0.0, 0.1), heading=(math.pi, math.pi)),
    )

    object_pose = mdp.UniformPose2dCommandCfg(
        asset_name="box_1",
        # asset_name="robot",
        simple_heading=False,
        resampling_time_range=(sampling_time, sampling_time),
        debug_vis=False,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            # pos_x=(-2.5, 2.5), pos_y=(-2.5, 2.5), heading=(-math.pi, math.pi)
            # pos_x=(-2.3, -2.3), pos_y=(-1.2, -1.2), heading=(0, 0)
            pos_x=(-0.0, -0.0), pos_y=(-0.0, -0.0), heading=(0, 0)
        ),
    )




@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        policy_path=f"/home/zhou/IsaacLab/logs/rsl_rl/unitree_go2_rough/walking_stable_100/exported/policy.pt",
        # policy_path=f"/home/zhou/IsaacLab/logs/rsl_rl/unitree_go2_flat/2024-07-21_16-03-10/exported/policy.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
        debug_vis=False,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        # object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("box_1")})
        object_position = ObsTerm(func=mdp.relative_object_position_in_world_frame, params={"object_cfg": SceneEntityCfg("box_1")})
        # object_position = ObsTerm(func=mdp.object_pos_w)
        # target_object_position = ObsTerm(func=mdp.generated_object_commands, params={"command_name": "object_pose"})
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        # target_object_position = ObsTerm(func=mdp.target_object_pos_w, params={"command_name": "object_pose"})

        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        # general obs
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

        object_size = ObsTerm(func=mdp.object_size_specific, params={"object_cfg": SceneEntityCfg("box_1")})

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))

        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # actions = ObsTerm(func=mdp.joint_pos)
        # actions = ObsTerm(func=mdp.last_action)
        actions = ObsTerm(func=mdp.last_action_joint_pos)
        # print("actions: ", actions)
        height_scan = ObsTerm(
            # func=mdp.height_scan,
            # params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            func=mdp.height_scan_with_object,
            # simulation2
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "object_cfg": SceneEntityCfg("box_1"), "object_size": (0.8, 0.8, 0.22)}, # 0.18 #0.8 , 0.65
            # simulation3
            # params={"sensor_cfg": SceneEntityCfg("height_scanner"), "object_cfg": SceneEntityCfg("box_2"), "object_size": (0.9, 0.6, 0.35)}, # 0.18 #0.8 , 0.65
            # simulation4
            # params={"sensor_cfg": SceneEntityCfg("height_scanner"), "object_cfg": SceneEntityCfg("box_1"), "object_size": (0.7, 0.9, 0.3)},
            # params={"sensor_cfg": SceneEntityCfg("height_scanner"), "object_cfg": SceneEntityCfg("box_2"), "object_size": (1.0, 0.58, 0.35)},
            # simulation5
            # params={"sensor_cfg": SceneEntityCfg("height_scanner"), "object_cfg": SceneEntityCfg("box_2"), "object_size": (0.8, 0.6, 0.3)},


            # func=mdp.height_scan_with_multiple_objects,
            # params={
            #     "sensor_cfg": SceneEntityCfg("height_scanner"), 
            #     "object_cfgs": [SceneEntityCfg("box_1"), SceneEntityCfg("box_2")], 
            #     "object_sizes": [(0.9, 0.9, 0.35), (0.9, 0.6, 0.18)]},
                # "object_cfgs": [SceneEntityCfg("box_1")], 
                # "object_sizes": [(0.7, 0.9, 0.3)]},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        # mode="startup",
        params={
            # simulation start
            # "pose_range": {"x": (-0., 0.), "y": (2., 2.), "yaw": (-1.57, -1.57)},
            # "pose_range": {"x": (-0., 0.), "y": (-2., -2.), "z": (0., 0.), "yaw": (-3.14, -3.14)},
            "pose_range": {"x": (-3.0, -3.0), "y": (-3.0, -3.0), "z": (0., 0.), "yaw": (0, 0)},
            # "pose_range": {"x": (-0.3, -0.3), "y": (-4.2, -4.2), "yaw": (3.0, 3.0)},
            # "pose_range": {"x": (0.6, 0.6), "y": (-1.2, -1.2), "yaw": (3.0, 3.0)},
            # "pose_range": {"x": (-1.3, -1.3), "y": (-3.6, -3.6), "yaw": (2.8, 2.8)},

            # final position with push red box
            # "pose_range": {"x": (-3.0, -3.0), "y": (-1.0, -1.0), "z": (0.42, 0.42), "yaw": (2.2, 2.2)},

            # climb to green box
            # "pose_range": {"x": (-3.5, -3.5), "y": (-2.3, -2.3), "z": (0.15, 0.15), "yaw": (1.57, 1.57)},

            # "pose_range": {"x": (0, 0), "y": (-1.3, -1.3), "yaw": (3.14, 3.14)},

            # final position with green box
            # "pose_range": {"x": (-3.2, -3.2), "y": (-1.0, -1.0), "z": (0.42, 0.42), "yaw": (2.2, 2.2)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_object_position_box1 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        # mode="startup",
        params={
            "pose_range": {"x": (0, 0), "y": (0, 0), "z": (0.0, 0.2)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("box_1"),
        },
    )

    reset_object_position_box2 = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        # mode="startup",
        params={
            "pose_range": {"x": (0, 0), "y": (0, 0), "z": (0.0, 0.2)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("box_2"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)

    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.0)

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 2.0, "command_name": "object_pose", "object_cfg": SceneEntityCfg("box_1")},
        weight=10.0,
    )

    # add this reward after several training
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 1.0, "command_name": "object_pose", "object_cfg": SceneEntityCfg("box_1")},
        weight=5.0,
    )

    positive_x_vel = RewTerm(
        func=mdp.positive_x_vel,
        weight=0.1, # 1 for commad[:, :3]
    )

    face_box = RewTerm(
        func=mdp.face_box,
        params={"object_cfg": SceneEntityCfg("box_1")},
        weight=2.0,
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )
    # foot_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"), "threshold": 1500.0},
    # )
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base"),"limit_angle": math.pi/2},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class GameEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = self.commands.object_pose.resampling_time_range[1]
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # self.decimation = self.decimation * 10
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.actions.pre_trained_policy_action.low_level_decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
