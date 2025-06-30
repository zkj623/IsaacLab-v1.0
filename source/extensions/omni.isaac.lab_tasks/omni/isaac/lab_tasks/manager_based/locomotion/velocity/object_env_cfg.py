# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
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
from omni.isaac.lab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR
import omni.isaac.lab.sim as sim_utils
from pxr import Gf, Sdf, Semantics, Usd, UsdGeom, Vt
import omni.isaac.core.utils.prims as prim_utils
import omni.usd

import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp
from omni.isaac.lab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
# import omni.isaac.lab_tasks.manager_based.navigation.mdp as mdp

##
# Pre-defined configs
##
from omni.isaac.lab.terrains.config.rough import ROUGH_TERRAINS_CFG, NAV_TERRAINS_CFG, OBJECT_PUSH_TERRAINS_CFG  # isort: skip
import numpy as np
import re
import random

LOW_LEVEL_ENV_CFG = UnitreeGo2RoughEnvCfg()

##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        # terrain_generator=ROUGH_TERRAINS_CFG,
        terrain_generator=OBJECT_PUSH_TERRAINS_CFG,
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
    # robots
    robot: ArticulationCfg = MISSING
    # target object
    # box_1: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=[0, 0, 0.2], rot=[1, 0, 0, 0]),
    #     spawn=sim_utils.CuboidCfg(
    #         # size=(0.2+0.6*random.random(), 0.2+0.6*random.random(), 0.2+0.2*random.random()),
    #         size=(0.8, 0.8, 0.18),
    #         rigid_props=RigidBodyPropertiesCfg(
    #             solver_position_iteration_count=16,
    #             solver_velocity_iteration_count=1,
    #             max_angular_velocity=1000.0,
    #             max_linear_velocity=1000.0,
    #             max_depenetration_velocity=5.0,
    #             disable_gravity=False,
    #         ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
    #         physics_material=sim_utils.RigidBodyMaterialCfg(),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.5)),
    #     ),
    #     debug_vis=False,
    # )
    box_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                # sim_utils.CuboidCfg(
                #     size=(0.4+0.6*random.random(), 0.4+0.6*random.random(), 0.2+0.2*random.random()),
                #     rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                #     # mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                #     mass_props=sim_utils.MassPropertiesCfg(density=700.0),
                #     collision_props=sim_utils.CollisionPropertiesCfg(),
                #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0), metallic=0.2),
                # ) for _ in range(100)
                ###########
                sim_utils.CuboidCfg(
                    size=(
                        0.3 + 0.6 * i / 10,
                        0.3 + 0.6 * j / 10,
                        0.2 + 0.2 * k / 10
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    # mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
                    mass_props=sim_utils.MassPropertiesCfg(density=50.0),
                    # mass_props=sim_utils.MassPropertiesCfg(density=10.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    # visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0), metallic=0.2),
                    visual_material=sim_utils.MdlFileCfg(
                        mdl_path=f"{NVIDIA_NUCLEUS_DIR}/Materials/Base/Wall_Board/Cardboard.mdl",
                        project_uvw=True,
                        texture_scale=(0.25, 0.25),
                    ),
                ) 
                for k in range(10)
                for j in range(10)
                for i in range(10)
                ##########
                # sim_utils.ConeCfg(
                #     radius=0.3,
                #     height=0.6,
                #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                # ),
                # sim_utils.CuboidCfg(
                #     size=(0.3, 0.3, 0.3),
                #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                # ),
                # sim_utils.SphereCfg(
                #     radius=0.3,
                #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                # ),
            ],
            random_choice=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            # mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.2), rot=[1, 0, 0, 0]),
    )
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
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


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # pose_command = mdp.TerrainBasedPose2dCommandCfg(
    #     asset_name="robot",
    #     simple_heading=True,
    #     resampling_time_range=(8.0, 8.0),
    #     debug_vis=False,
    #     ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-1.5, -1.5), pos_y=(-0.0, 0.0), heading=(-math.pi, math.pi)),
    # )

    object_pose = mdp.UniformPose2dCommandCfg(
        asset_name="box_1",
        # asset_name="robot",
        simple_heading=False,
        resampling_time_range=(12.0, 12.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(
            pos_x=(-3, 3), pos_y=(-3, 3), heading=(0, 0)
            # for display
            # pos_x=(2, 2), pos_y=(-1.5, -1.5), heading=(0, 0)
            # pos_x=(-2.0, -2.0), pos_y=(-1.2, -1.2), heading=(-math.pi, math.pi)
        ),
    )



@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)

    pre_trained_policy_action: mdp.PreTrainedPolicyActionCfg = mdp.PreTrainedPolicyActionCfg(
        asset_name="robot",
        # policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/Blind/policy.pt",
        policy_path=f"/home/zhou/IsaacLab/logs/rsl_rl/unitree_go2_rough/walking_stable_100/exported/policy.pt",
        # policy_path=f"/home/zhou/IsaacLab/logs/rsl_rl/unitree_go2_flat/2024-07-21_16-03-10/exported/policy.pt",
        low_level_decimation=4,
        low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
        low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        # robot_position = ObsTerm(func=mdp.robot_pos_w)
        # object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame, params={"object_cfg": SceneEntityCfg("box_1")})
        object_position = ObsTerm(func=mdp.relative_object_position_in_world_frame, params={"object_cfg": SceneEntityCfg("box_1")})
        # object_position = ObsTerm(func=mdp.object_pos_w)
        # target_object_position = ObsTerm(func=mdp.generated_object_commands, params={"command_name": "object_pose"})
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        # target_object_position = ObsTerm(func=mdp.target_object_pos_w, params={"command_name": "object_pose"})

        projected_gravity = ObsTerm(func=mdp.projected_gravity)

        object_size = ObsTerm(func=mdp.object_size, params={"object_cfg": SceneEntityCfg("box_1")})

        # general obs
        # pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})

        # actions = ObsTerm(func=mdp.last_action)

        # joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )

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
        params={
            "pose_range": {"x": (-0., 0.), "y": (-0., 0.), "yaw": (-3.14, 3.14)},
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

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0, 0), "y": (0, 0), "z": (0.0, 0.2)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("box_1"),
        },
    )

    # object_scale_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("box_1"),
    #         "mass_distribution_params": (1.0, 10.0),
    #         "operation": "abs",
    #     },
    # )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    # track_lin_vel_xy_exp = RewTerm(
    #     func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )
    # track_ang_vel_z_exp = RewTerm(
    #     func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    # )
    # # -- penalties
    # lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    # ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # dof_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    # dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    # dof_vel_limit = RewTerm(func=mdp.joint_vel_limits, weight=-1, params={"soft_ratio": 1})
    # dof_torques_limit = RewTerm(func=mdp.applied_torque_limits, weight=-0.2)
    # base_acc = RewTerm(func=mdp.body_lin_acc_l2, weight=-0.001)

    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=0.125,
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT"),
    #         "command_name": "pose_command",
    #         "threshold": 0.5,
    #     },
    # )
    # undesired_contacts = RewTerm(
    #     func=mdp.undesired_contacts,
    #     weight=-1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*THIGH"), "threshold": 1.0},
    # )

    # -- optional penalties
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.0)

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-400.0)

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 1.0, "command_name": "object_pose", "object_cfg": SceneEntityCfg("box_1")},
        weight=2.0,
    )

    object_ang_tracking = RewTerm(
        func=mdp.object_goal_angle,
        params={"std": 1.0, "command_name": "object_pose", "object_cfg": SceneEntityCfg("box_1")},
        weight=1.0,
    )

    # add this reward after several training
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.5, "command_name": "object_pose", "object_cfg": SceneEntityCfg("box_1")},
        weight=2.0,
    )

    # object_ang_tracking_fine_grained = RewTerm(
    #     func=mdp.object_goal_angle,
    #     params={"std": 0.25, "command_name": "object_pose", "object_cfg": SceneEntityCfg("box_1")},
    #     weight=1.0,
    # )

    positive_x_vel = RewTerm(
        func=mdp.positive_x_vel,
        weight=0.1, # 1 for commad[:, :3]
    )

    # object_vel_penalty = RewTerm(
    #     func=mdp.object_vel_penalty,
    #     params={"object_cfg": SceneEntityCfg("box_1")},
    #     weight=-5,
    # )

    # face_box = RewTerm(
    #     func=mdp.face_box,
    #     params={"object_cfg": SceneEntityCfg("box_1")},
    #     weight=2.0,
    # )

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
class PushingObjectFlatEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
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
        self.decimation = self.decimation * 10
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
