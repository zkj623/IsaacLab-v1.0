# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.terrains import FlatPatchSamplingCfg

from ..terrain_generator_cfg import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0., obstacle_height_mode='fixed', obstacle_height_range=(0.36, 0.36), obstacle_width_range=(1.4, 1.5), num_obstacles=1, platform_width=3.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.4, max_height_diff=0.05)
            }
        ),
        "high_obstacles": terrain_gen.HfHighObstaclesTerrainCfg(
            proportion=0., obstacle_height_mode='fixed', obstacle_height_range=(0.2, 0.3), obstacle_width_range=(1.4, 1.5), platform_width=3.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, x_range = (-3.5,-2.5), y_range = (-1.5, -1.5), patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "rails": terrain_gen.MeshRailsTerrainCfg(
            proportion=0., rail_thickness_range=(0.2,0.2), rail_height_range=(0.2,0.2), platform_width=2.0
        ),
        "pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0., pit_depth_range=(0.3,0.3), platform_width=3.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "gap": terrain_gen.MeshGapTerrainCfg(
            proportion=0., gap_width_range=(0.15,0.3), platform_width=3.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
    },
)
"""Rough terrains configuration."""

CLIMB_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0., grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0., noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "step_up": terrain_gen.MeshStepTerrainCfg(
            proportion=0., 
            step_height_range=(0.3, 0.4), # (0.35, 0.40) #0.55
            step_width=0.5,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
            up=True,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, x_range = (3.0, 3.0), y_range = (0.0, 0.0), patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "step_down": terrain_gen.MeshStepTerrainCfg(
            proportion=0., 
            step_height_range=(0.3, 0.4),
            step_width=0.5,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
            up=False,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, x_range = (3.0, 3.0), y_range = (0.0, 0.0), patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "step": terrain_gen.MeshSingleStepTerrainCfg(
            proportion=1., 
            step_height_range=(0.6, 0.6),
            step_length_range=(1.0, 2.0),
            step_width=0.5,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, x_range = (3.5, 3.5), y_range = (0.0, 0.0), patch_radius=0.3, max_height_diff=0.05)
            }
        ),
    },
)
"""climbing terrains configuration."""

NAV_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0., grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        # obstacles group
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=1., obstacle_height_mode='fixed', obstacle_height_range=(0.2, 0.6), obstacle_width_range=(0.5, 1.5), num_obstacles=20, platform_width=2.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, z_range=(0, 0.2), patch_radius=0.4, max_height_diff=0.05)
            }
        ),
        "high_obstacles": terrain_gen.HfHighObstaclesTerrainCfg(
            proportion=0., obstacle_height_mode='fixed', obstacle_height_range=(0.2, 0.3), obstacle_width_range=(1.4, 1.5), platform_width=3.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, x_range = (-3.5,-2.5), y_range = (-1.5, -1.5), patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0., noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        # navigation example in the paper
        "step": terrain_gen.MeshSingleStepTerrainCfg(
            proportion=0., 
            step_height_range=(0.6, 0.6),
            step_length_range=(1.0, 2.0),
            step_width=0.5,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, x_range = (3.5, 3.5), y_range = (0.0, 0.0), patch_radius=0.3, max_height_diff=0.05)
            }
        ),
    },
)

OBJECT_PUSH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0., grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0., obstacle_height_mode='fixed', obstacle_height_range=(0.2, 0.6), obstacle_width_range=(0.8, 1.0), num_obstacles=10, platform_width=2,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, x_range = (-4,4), y_range = (-4, 4), z_range=(0, 0.2), patch_radius=0.4, max_height_diff=0.05)
            }
        ),
        "high_obstacles": terrain_gen.HfHighObstaclesTerrainCfg(
            proportion=0., obstacle_height_mode='fixed', obstacle_height_range=(0.2, 0.3), obstacle_width_range=(1.4, 1.5), platform_width=3.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, x_range = (-3.5,-2.5), y_range = (-1.5, -1.5), patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.8, noise_range=(0.01, 0.05), noise_step=0.02, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.005, 0.01), noise_step=0.005, border_width=3.8,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
    },
)

INTERACTION_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0., grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0., obstacle_height_mode='fixed', obstacle_height_range=(0.2, 0.6), obstacle_width_range=(0.8, 1.0), num_obstacles=10, platform_width=2,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, x_range = (-4,4), y_range = (-4, 4), z_range=(0, 0.2), patch_radius=0.4, max_height_diff=0.05)
            }
        ),
        "high_obstacles": terrain_gen.HfHighObstaclesTerrainCfg(
            proportion=0., obstacle_height_mode='fixed', obstacle_height_range=(0.2, 0.3), obstacle_width_range=(1.4, 1.5), platform_width=3.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, x_range = (-3.5,-2.5), y_range = (-1.5, -1.5), patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0., noise_range=(0.01, 0.05), noise_step=0.02, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "flat": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0., noise_range=(0.005, 0.01), noise_step=0.005, border_width=3.8,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        # "wall": terrain_gen.MeshWallTerrainCfg(
        #     proportion=0., wall_height_range=1.0, wall_width_range=(1.4, 1.5), platform_width=2.0,
        #     flat_patch_sampling={
        #         "target": FlatPatchSamplingCfg(num_patches=30, x_range = (-3.5,-2.5), y_range = (-1.5, -1.5), patch_radius=0.3, max_height_diff=0.05)
        #     }
        # ),
        "step_up": terrain_gen.MeshStepTerrainCfg(
            proportion=0., 
            step_height_range=(0.3, 0.4), # (0.35, 0.40) #0.55
            step_width=0.5,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
            up=True,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, x_range = (3.0, 3.0), y_range = (0.0, 0.0), patch_radius=0.3, max_height_diff=0.05)
            }
        ),
    },
)


"""navigation terrains configuration."""


TEST_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        # obstacles group
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.5, obstacle_height_mode='fixed', obstacle_height_range=(0.4, 0.6), obstacle_width_range=(0.5, 1.5), num_obstacles=20, platform_width=2.0,
            # flat_patch_sampling={
            #     "target": FlatPatchSamplingCfg(num_patches=30, z_range=(0, 0.2), patch_radius=0.4, max_height_diff=0.05)
            # }
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.2, 0.4), platform_width=2.0, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.2, 0.4), platform_width=2.0, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
    },
)