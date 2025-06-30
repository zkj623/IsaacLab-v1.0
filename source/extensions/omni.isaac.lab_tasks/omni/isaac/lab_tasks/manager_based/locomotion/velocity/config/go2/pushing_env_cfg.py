# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.manager_based.locomotion.velocity.object_env_cfg import PushingObjectFlatEnvCfg

##
# Pre-defined configs
##
from omni.isaac.lab_assets.unitree import UNITREE_GO2_CFG  # isort: skip
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, MassPropertiesCfg
from omni.isaac.lab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR, NVIDIA_NUCLEUS_DIR


@configclass
class UnitreeGo2PushingFlatEnvCfg(PushingObjectFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base"
        
        self.scene.env_spacing = 5

        # change terrain to flat
        # annotate the following line to train pushing policy in environment with obstacles
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        self.curriculum.terrain_levels = None

        # reduce action scale
        # self.actions.joint_pos.scale = 0.25

        # event
        self.events.reset_base.params = {
            # randomize the initial position of the robot
            "pose_range": {"x": (-3., 3.), "y": (-3., 3.), "z": (0.0, 0.0), "yaw": (0, 6.28)},
            # set the initial position of the robot
            # "pose_range": {"x": (-0., -0.), "y": (2., 2.), "z": (0.0, 0.0), "yaw": (-1.57, -1.57)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (-0.0, -0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.reset_robot_joints.params["position_range"] = (1.0, 1.0)

        # velocity rewards
        # self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        # self.rewards.feet_air_time.weight = 0.01
        # self.rewards.undesired_contacts = None
        # self.rewards.dof_torques_l2.weight = -0.0002
        # self.rewards.dof_acc_l2.weight = -2.5e-7
        # self.rewards.track_lin_vel_xy_exp.weight = 1.5
        # self.rewards.track_ang_vel_z_exp.weight = 0.75

        # position rewards
        # self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_calf"
        # self.rewards.dof_torques_l2.weight = -0.0002
        # self.rewards.dof_acc_l2.weight = -2.5e-7


        # terminations
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base"


@configclass
class UnitreeGo2PushingFlatEnvCfg_PLAY(UnitreeGo2PushingFlatEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        # self.episode_length_s = 8.0
        self.scene.num_envs = 50
        self.scene.env_spacing = 5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
