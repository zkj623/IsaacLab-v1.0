# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import omni.isaac.lab.terrains as terrain_gen
from omni.isaac.lab.terrains import FlatPatchSamplingCfg

from ..terrain_generator_cfg import TerrainGeneratorCfg


NAVIGATE_ANYWHERE_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "high_obstacles": terrain_gen.HfHighObstaclesTerrainCfg(
            proportion=1., obstacle_height_mode='fixed', obstacle_height_range=(0.35, 0.35), obstacle_width_range=(1.49, 1.5), platform_width=3.0,
            flat_patch_sampling={
                # "target": FlatPatchSamplingCfg(num_patches=30, x_range = (-3.5, -3.5), y_range = (-1.2, -1.2), patch_radius=0.3, max_height_diff=0.05)
                "target": FlatPatchSamplingCfg(num_patches=30, x_range = (-3.5, -3.5), y_range = (-1.2, -1.2), patch_radius=0.3, max_height_diff=0.05)
                # "target": FlatPatchSamplingCfg(num_patches=30, x_range = (0, 0), y_range = (-1.2, -1.2), patch_radius=0.3, max_height_diff=0.05)
            }
        ),
    },
)

SIMULATION1_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "high_obstacles": terrain_gen.HfHighObstaclesTerrainCfg(
            proportion=1., obstacle_height_mode='fixed', obstacle_height_range=(0.35, 0.35), obstacle_width_range=(1.49, 1.5), platform_width=3.0,
            # flat_patch_sampling={
            #     # "target": FlatPatchSamplingCfg(num_patches=30, x_range = (-3.5, -3.5), y_range = (-1.2, -1.2), patch_radius=0.3, max_height_diff=0.05)
            #     "target": FlatPatchSamplingCfg(num_patches=30, x_range = (-3.5, -3.5), y_range = (-1.2, -1.2), patch_radius=0.3, max_height_diff=0.05)
            #     # "target": FlatPatchSamplingCfg(num_patches=30, x_range = (0, 0), y_range = (-1.2, -1.2), patch_radius=0.3, max_height_diff=0.05)
            # }
        ),
    },
)

NAVIGATE_ANYWHERE2_CFG = TerrainGeneratorCfg(
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
            proportion=0.,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0., grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=1., obstacle_height_mode='fixed', obstacle_height_range=(0.36, 0.36), obstacle_width_range=(1.4, 1.5), num_obstacles=1, platform_width=3.0,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.4, max_height_diff=0.05)
            }
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0., noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0., slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0., slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
    },
)

PIT_CROSS_CFG = TerrainGeneratorCfg(
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
            proportion=0.,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0., grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0,
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
        "pit_cross": terrain_gen.MeshPitCrossTerrainCfg(
            proportion=1., pit_depth=0.6, platform_width=2.0,
            flat_patch_sampling={
                # "target": FlatPatchSamplingCfg(num_patches=30, x_range = (-3.5, -3.5), y_range = (-1.2, -1.2), patch_radius=0.3, max_height_diff=0.05)
                "target": FlatPatchSamplingCfg(num_patches=30, x_range = (-3.5, -3.5), y_range = (-0.7, -0.7), patch_radius=0.2, max_height_diff=0.05)
                # "target": FlatPatchSamplingCfg(num_patches=30, x_range = (0, 0), y_range = (-1.2, -1.2), patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "pit_cross": terrain_gen.MeshPitCrossTerrainCfg(
            proportion=0., pit_depth=0.6, platform_width=2.0,
            flat_patch_sampling={
                # "target": FlatPatchSamplingCfg(num_patches=30, x_range = (-3.5, -3.5), y_range = (-1.2, -1.2), patch_radius=0.3, max_height_diff=0.05)
                "target": FlatPatchSamplingCfg(num_patches=30, x_range = (-3.5, -3.5), y_range = (-0.7, -0.7), patch_radius=0.2, max_height_diff=0.05)
                # "target": FlatPatchSamplingCfg(num_patches=30, x_range = (0, 0), y_range = (-1.2, -1.2), patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "corridor": terrain_gen.MeshCorridorTerrainCfg(
            proportion=1., platform_width=2.0,
            flat_patch_sampling={
                # "target": FlatPatchSamplingCfg(num_patches=30, x_range = (-3.5, -3.5), y_range = (-1.2, -1.2), patch_radius=0.3, max_height_diff=0.05)
                "target": FlatPatchSamplingCfg(num_patches=30, x_range = (-3.5, -3.5), y_range = (-0.7, -0.7), patch_radius=0.2, max_height_diff=0.05)
                # "target": FlatPatchSamplingCfg(num_patches=30, x_range = (0, 0), y_range = (-1.2, -1.2), patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0., slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0., slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(num_patches=30, patch_radius=0.3, max_height_diff=0.05)
            }
        ),
    },
)


"""Rough terrains configuration."""