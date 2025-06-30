# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper to configure an :class:`ManagerBasedRLEnv` instance to RSL-RL vectorized environment.

The following example shows how to wrap an environment for RSL-RL:

.. code-block:: python

    from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper

    env = RslRlVecEnvWrapper(env)

"""


import gymnasium as gym
import torch
import numpy as np
import copy

from rsl_rl.env import VecEnv

from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv
from omni.isaac.lab.managers import ObservationManager, ActionManager, EventManager, TerminationManager
from omni.isaac.lab.managers import ObservationTermCfg as ObsTerm
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import omni.isaac.lab_tasks.manager_based.locomotion.velocity.mdp as mdp

class RslRlVecEnvWrapper(VecEnv):
    """Wraps around Isaac Lab environment for RSL-RL library

    To use asymmetric actor-critic, the environment instance must have the attributes :attr:`num_privileged_obs` (int).
    This is used by the learning agent to allocate buffers in the trajectory memory. Additionally, the returned
    observations should have the key "critic" which corresponds to the privileged observations. Since this is
    optional for some environments, the wrapper checks if these attributes exist. If they don't then the wrapper
    defaults to zero as number of privileged observations.

    .. caution::

        This class must be the last wrapper in the wrapper chain. This is because the wrapper does not follow
        the :class:`gym.Wrapper` interface. Any subsequent wrappers will need to be modified to work with this
        wrapper.

    Reference:
        https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/env/vec_env.py
    """

    def __init__(self, env: ManagerBasedRLEnv):
        """Initializes the wrapper.

        Note:
            The wrapper calls :meth:`reset` at the start since the RSL-RL runner does not call reset.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        # store information required by wrapper
        self.num_envs = self.unwrapped.num_envs
        self.device = self.unwrapped.device
        self.max_episode_length = self.unwrapped.max_episode_length
        if hasattr(self.unwrapped, "action_manager"):
            self.num_actions = self.unwrapped.action_manager.total_action_dim
        else:
            self.num_actions = self.unwrapped.num_actions
        if hasattr(self.unwrapped, "observation_manager"):
            self.num_obs = self.unwrapped.observation_manager.group_obs_dim["policy"][0]
        else:
            self.num_obs = self.unwrapped.num_observations
        # -- privileged observations
        if (
            hasattr(self.unwrapped, "observation_manager")
            and "critic" in self.unwrapped.observation_manager.group_obs_dim
        ):
            self.num_privileged_obs = self.unwrapped.observation_manager.group_obs_dim["critic"][0]
        elif hasattr(self.unwrapped, "num_states"):
            self.num_privileged_obs = self.unwrapped.num_states
        else:
            self.num_privileged_obs = 0
        # reset at the start since the RSL-RL runner does not call reset
        self.env.reset()

    def __str__(self):
        """Returns the wrapper name and the :attr:`env` representation string."""
        return f"<{type(self).__name__}{self.env}>"

    def __repr__(self):
        """Returns the string representation of the wrapper."""
        return str(self)

    """
    Properties -- Gym.Wrapper
    """

    @property
    def cfg(self) -> object:
        """Returns the configuration class instance of the environment."""
        return self.unwrapped.cfg

    @property
    def render_mode(self) -> str | None:
        """Returns the :attr:`Env` :attr:`render_mode`."""
        return self.env.render_mode

    @property
    def observation_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`observation_space`."""
        return self.env.observation_space

    @property
    def action_space(self) -> gym.Space:
        """Returns the :attr:`Env` :attr:`action_space`."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Returns the class name of the wrapper."""
        return cls.__name__

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        """Returns the base environment of the wrapper.

        This will be the bare :class:`gymnasium.Env` environment, underneath all layers of wrappers.
        """
        return self.env.unwrapped

    """
    Properties
    """

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Returns the current observations of the environment."""
        if hasattr(self.unwrapped, "observation_manager"):
            obs_dict = self.unwrapped.observation_manager.compute()
        else:
            obs_dict = self.unwrapped._get_observations()
        return obs_dict["policy"], {"observations": obs_dict}

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """The episode length buffer."""
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        """Set the episode length buffer.

        Note:
            This is needed to perform random initialization of episode lengths in RSL-RL.
        """
        self.unwrapped.episode_length_buf = value

    """
    Operations - MDP
    """

    def seed(self, seed: int = -1) -> int:  # noqa: D102
        return self.unwrapped.seed(seed)

    def reset(self) -> tuple[torch.Tensor, dict]:  # noqa: D102
        # reset the environment
        obs_dict, _ = self.env.reset()
        # return observations
        return obs_dict["policy"], {"observations": obs_dict}

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        obs = obs_dict["policy"]
        extras["observations"] = obs_dict
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # return the step information
        return obs, rew, dones, extras

    def close(self):  # noqa: D102
        return self.env.close()


class PushEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = {'policy': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9, 16), dtype=np.float32)}
        pushing_obs_cfg = copy.deepcopy(self.unwrapped.cfg.observations)
        pushing_obs_cfg.policy.pose_command = None
        pushing_obs_cfg.policy.base_ang_vel = None
        pushing_obs_cfg.policy.joint_pos = None
        pushing_obs_cfg.policy.joint_vel = None
        pushing_obs_cfg.policy.actions = None
        pushing_obs_cfg.policy.height_scan = None
        # print(pushing_obs_cfg.policy)
        self.unwrapped.observation_manager = ObservationManager(pushing_obs_cfg, self.unwrapped)

        walk_action_cfg = copy.deepcopy(self.unwrapped.cfg.actions)
        walk_action_cfg.joint_pos = None

        self.unwrapped.action_manager = ActionManager(walk_action_cfg, self.unwrapped)

    def observation(self, obs):
        # obs_tensor = obs["policy"][:, :13]
        obs_tensor = torch.cat((obs["policy"][:, :13], obs["policy"][:, 17:20]), dim=1)
        return {'policy': obs_tensor}
    
    # def action(self, action):
    #     # obs_tensor = obs["policy"][:, :13]
    #     action_tensor = action[:, 12:15]
    #     return action_tensor
    
    # def step(self, action):
    #     return self.env.step(self.action(action))
    

class WalkEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = {'policy': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9, 10), dtype=np.float32)}
        walk_obs_cfg = copy.deepcopy(self.unwrapped.cfg.observations)
        # remember to change the observation space
        walk_obs_cfg.policy.object_position = None
        walk_obs_cfg.policy.target_object_position = None
        walk_obs_cfg.policy.object_size = None
        walk_obs_cfg.policy.base_ang_vel = None
        walk_obs_cfg.policy.joint_pos = None
        walk_obs_cfg.policy.joint_vel = None
        walk_obs_cfg.policy.actions = None
        walk_obs_cfg.policy.height_scan = None

        self.unwrapped.observation_manager = ObservationManager(walk_obs_cfg, self.unwrapped)

        walk_action_cfg = copy.deepcopy(self.unwrapped.cfg.actions)
        walk_action_cfg.joint_pos = None

        self.unwrapped.action_manager = ActionManager(walk_action_cfg, self.unwrapped)

    def observation(self, obs):
        obs_tensor = torch.cat((obs["policy"][:, :3], obs["policy"][:, 10:17]), dim=1)
        return {'policy': obs_tensor}
    
class ClimbEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = {'policy': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9, 236), dtype=np.float32)}
        climb_obs_cfg = copy.deepcopy(self.unwrapped.cfg.observations)
        # remember to change the observation space
        climb_obs_cfg.policy.object_position = None
        climb_obs_cfg.policy.target_object_position = None
        climb_obs_cfg.policy.object_size = None

        self.unwrapped.observation_manager = ObservationManager(climb_obs_cfg, self.unwrapped)

        walk_action_cfg = copy.deepcopy(self.unwrapped.cfg.actions)
        walk_action_cfg.pre_trained_policy_action = None

        self.unwrapped.action_manager = ActionManager(walk_action_cfg, self.unwrapped)
        
    def observation(self, obs):
        obs_tensor = torch.cat((obs["policy"][:, :3], obs["policy"][:, 20:23], obs["policy"][:, 10:17], obs["policy"][:, 23:246]), dim=1)
        return {'policy': obs_tensor}
        # return obs
    
class NavEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = {'policy': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9, 197), dtype=np.float32)}
        nav_obs_cfg = copy.deepcopy(self.unwrapped.cfg.observations)
        # remember to change the observation space
        nav_obs_cfg.policy.object_position = None
        nav_obs_cfg.policy.target_object_position = None
        nav_obs_cfg.policy.object_size = None
        nav_obs_cfg.policy.base_ang_vel = None
        nav_obs_cfg.policy.joint_pos = None
        nav_obs_cfg.policy.joint_vel = None
        nav_obs_cfg.policy.actions = None

        self.unwrapped.observation_manager = ObservationManager(nav_obs_cfg, self.unwrapped)

    def observation(self, obs):
        obs_tensor = torch.cat((obs["policy"][:, :3], obs["policy"][:, 10:17], obs["policy"][:, 59:246]), dim=1)
        return {'policy': obs_tensor}
        # return obs

# class ObjectClimbEnvWrapper(gym.ObservationWrapper):
#     def __init__(self, env):
#         super().__init__(env)
#         walk_action_cfg = copy.deepcopy(self.unwrapped.cfg.actions)
#         walk_action_cfg.pre_trained_policy_action.low_level_observations.height_scan = ObsTerm(
#             func=mdp.height_scan_with_object,
#             params={"sensor_cfg": SceneEntityCfg("height_scanner")},
#             noise=Unoise(n_min=-0.1, n_max=0.1),
#             clip=(-1.0, 1.0),
#         )
#         self.unwrapped.action_manager = ActionManager(walk_action_cfg, self.unwrapped)

#     def observation(self, obs):
#         return obs


class OriginalEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.observation_space = {'policy': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9, 12), dtype=np.float32)}
        self.unwrapped.observation_manager = ObservationManager(self.unwrapped.cfg.observations, self.unwrapped)
        
        walk_action_cfg = copy.deepcopy(self.unwrapped.cfg.actions)
        walk_action_cfg.joint_pos = None
        # walk_action_cfg.pre_trained_policy_action = None

        self.unwrapped.action_manager = ActionManager(walk_action_cfg, self.unwrapped)

        event_cfg = copy.deepcopy(self.unwrapped.cfg.events)
        event_cfg.reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                # simulation 1 & 2 & 3
                "pose_range": {"x": (-3.0, -3.0), "y": (-3.0, -3.0), "z": (-0., -0.), "yaw": (0, 0)},
                # simulation 4
                # "pose_range": {"x": (-3.0, -3.0), "y": (-3.0, -3.0), "z": (-0.6, -0.6), "yaw": (0, 0)},
                # "pose_range": {"x": (1.0, 1.0), "y": (-0.5, -0.5), "z": (-0., -0.), "yaw": (0, 0)},
                
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

        event_cfg.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (1.0, 1.0),
                "velocity_range": (0.0, 0.0),
            },
        )

        event_cfg.reset_object_position_box1 = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                # simulation 1 2 3
                "pose_range": {"x": (0, 0), "y": (0, 0), "z": (-0.1, -0.1)},
                # simulation 4
                # "pose_range": {"x": (0, 0), "y": (0, 0), "z": (-0.4, -0.4)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("box_1"),
            },
        )

        event_cfg.reset_object_position_box2 = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                # simulation 1 2 3
                "pose_range": {"x": (0, 0), "y": (0, 0), "z": (-0.1, -0.1)},
                # simulation 4
                # "pose_range": {"x": (0, 0), "y": (0, 0), "z": (-0.5, -0.5)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("box_2"),
            },
        )

        event_cfg.reset_object_position_box3 = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (0, 0), "y": (0, 0), "z": (-0.1, -0.1)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("box_3"),
            },
        )

        self.unwrapped.event_manager = EventManager(event_cfg, self.unwrapped)

    def observation(self, obs):
        return obs
    
    # def action(self, action):
    #     print("raw_action:", action)
    #     action_tensor = action[:, 12:15]
    #     return action_tensor
    
    # def step(self, action):
    #     return self.env.step(self.action(action))

class OriginalHighEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.observation_space = {'policy': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9, 12), dtype=np.float32)}
        self.unwrapped.observation_manager = ObservationManager(self.unwrapped.cfg.observations, self.unwrapped)
        
        walk_action_cfg = copy.deepcopy(self.unwrapped.cfg.actions)
        walk_action_cfg.joint_pos = None
        # walk_action_cfg.pre_trained_policy_action = None

        self.unwrapped.action_manager = ActionManager(walk_action_cfg, self.unwrapped)

        event_cfg = copy.deepcopy(self.unwrapped.cfg.events)
        event_cfg.reset_base = None
        event_cfg.reset_robot_joints = None
        event_cfg.reset_object_position_box1 = None
        event_cfg.reset_object_position_box2 = None
        event_cfg.reset_object_position_box3 = None

        self.unwrapped.event_manager = EventManager(event_cfg, self.unwrapped)

    def observation(self, obs):
        return obs
    
class OriginalLowEnvWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # self.observation_space = {'policy': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(9, 12), dtype=np.float32)}
        self.unwrapped.observation_manager = ObservationManager(self.unwrapped.cfg.observations, self.unwrapped)
        
        walk_action_cfg = copy.deepcopy(self.unwrapped.cfg.actions)
        # walk_action_cfg.joint_pos = None
        walk_action_cfg.pre_trained_policy_action = None

        self.unwrapped.action_manager = ActionManager(walk_action_cfg, self.unwrapped)

        event_cfg = copy.deepcopy(self.unwrapped.cfg.events)
        event_cfg.reset_base = None
        event_cfg.reset_robot_joints = None
        event_cfg.reset_object_position_box1 = None
        event_cfg.reset_object_position_box2 = None
        event_cfg.reset_object_position_box3 = None

        self.unwrapped.event_manager = EventManager(event_cfg, self.unwrapped)

        termination_cfg = copy.deepcopy(self.unwrapped.cfg.terminations)
        termination_cfg.base_contact = None
        termination_cfg.bad_orientation = None
        self.unwrapped.termination_manager = TerminationManager(termination_cfg, self.unwrapped)

    def observation(self, obs):
        return obs