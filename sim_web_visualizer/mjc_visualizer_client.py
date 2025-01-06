# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the MIT License [see LICENSE for details].

from collections import namedtuple
from functools import partial
from pathlib import Path
from typing import Optional, Callable, List, Union, Dict

import meshcat
import meshcat.geometry as g
import numpy as np
import quaternion
import torch
import transforms3d
import mujoco_py
from numpy.lib.recfunctions import structured_to_unstructured

from sim_web_visualizer.base_visualizer_client import MeshCatVisualizerBase, AssetResource
from sim_web_visualizer.utils.rotation_utils import compute_vector_rotation

USE_GPU_PIPELINE = False


class MimicSim:
    def __init__(self, sim: mujoco_py.MjSim):
        self._sim = sim
        self.methods: Dict[str, Callable] = {}

    def add_method(self, name: str, fn: Callable):
        self.methods[name] = fn

    def __getattribute__(self, item):
        if item in ["add_method", "methods", "_sim"]:
            return super().__getattribute__(item)
        if item in self.methods:
            return self.methods[item]
        else:
            return self._sim.__getattribute__(item)


def set_env_pose(
    num_env: int, num_per_row: int, env_size: np.ndarray, scene_offset: np.ndarray, viz: meshcat.Visualizer
):
    row = num_env // num_per_row
    column = num_env % num_per_row
    env_origin_xy = env_size[:2] * np.array([column, row])
    env_pose = np.eye(4)
    env_pose[:2, 3] = env_origin_xy - scene_offset
    viz[f"/Sim/env:{num_env}"].set_transform(env_pose)


class MeshCatVisualizerMujoco(MeshCatVisualizerBase):
    def __init__(
        self,
        port: Optional[int] = None,
        host="localhost",
        keep_default_viewer=True,
        scene_offset=np.array([10.0, 10.0]),
        max_env=4,
    ):
        super().__init__(port, host)
        self.keep_default_viewer = keep_default_viewer

        self.original_sim: Optional[mujoco_py.MjSim] = None
        self.new_sim: Optional[MimicSim] = None
        self.max_env = max_env
        print(
            f"For efficiency, the sim_web_visualizer will only visualize the first {max_env} environments.\n"
            f"Modify the max_env parameter if you prefer more or less visualization."
        )

        # Cache
        self.asset_resource_map: Dict[int, AssetResource] = {}

    def set_sim_instance(self, sim: mujoco_py.MjSim, model_xml: str) -> MimicSim:
        self.original_sim = sim
        self.new_sim = MimicSim(self.original_sim)

        self.load_model(model_xml)
        self._override_step_fn()
        self._override_forward_fn()
        self._override_reset_fn()
        return self.new_sim

    def _override_step_fn(self):
        def step():
            self.original_sim.step()

            model = self.original_sim.model
            data = self.original_sim.data

            sim_viz = self.viz["/Sim"]
            env_i = 0
            env_viz = sim_viz[f"env:{env_i}"]
            for body_id in range(model.nbody):
                body_name = model.body_id2name(body_id)

                pos = data.body_xpos[body_id]
                rot = data.body_xmat[body_id].reshape(3, 3)

                T = np.eye(4)
                T[:3, :3] = rot
                T[:3, 3] = pos

                env_viz[body_name].set_transform(T)

        self.new_sim.add_method("step", step)

    def _override_forward_fn(self):
        def forward():
            self.original_sim.forward()

            model = self.original_sim.model
            data = self.original_sim.data

            sim_viz = self.viz["/Sim"]
            env_i = 0
            env_viz = sim_viz[f"env:{env_i}"]
            for body_id in range(model.nbody):
                body_name = model.body_id2name(body_id)

                pos = data.body_xpos[body_id]
                rot = data.body_xmat[body_id].reshape(3, 3)

                T = np.eye(4)
                T[:3, :3] = rot
                T[:3, 3] = pos

                env_viz[body_name].set_transform(T)

        self.new_sim.add_method("forward", forward)

    def _override_reset_fn(self):
        def reset():
            self.original_sim.reset()

            model = self.original_sim.model
            data = self.original_sim.data

            sim_viz = self.viz["/Sim"]
            env_i = 0
            env_viz = sim_viz[f"env:{env_i}"]
            for body_id in range(model.nbody):
                body_name = model.body_id2name(body_id)

                pos = data.body_xpos[body_id]
                rot = data.body_xmat[body_id].reshape(3, 3)

                T = np.eye(4)
                T[:3, :3] = rot
                T[:3, 3] = pos

                env_viz[body_name].set_transform(T)

        self.new_sim.add_method("reset", reset)

    def load_model(self, model_xml: str):
        resource = self.dry_load_asset(
            model_xml,
            collapse_fixed_joints=False,
            replace_cylinder_with_capsule=False,
            use_mesh_materials=True,
        )
        num_env = 0
        robot_tree_path = f"/Sim/env:{num_env}"
        self.load_asset_resources(resource, robot_tree_path, scale=1.0)
        self.asset_resource_map[0] = resource


_REGISTERED_VISUALIZER: List[MeshCatVisualizerMujoco] = []


def create_mjc_visualizer(port=None, host="localhost", keep_default_viewer=True, max_env=4, **kwargs):
    visualizer = MeshCatVisualizerMujoco(port, host, keep_default_viewer, max_env=max_env, **kwargs)
    if len(_REGISTERED_VISUALIZER) > 0:
        raise RuntimeError(f"You can only create a web visualizer once")
    visualizer.delete_all()
    _REGISTERED_VISUALIZER.append(visualizer)
    return visualizer


def bind_visualizer_to_sim(sim: mujoco_py.MjSim, model_xml: str) -> MimicSim:
    if len(_REGISTERED_VISUALIZER) <= 0:
        raise RuntimeError(f"Web Visualizer has not been created yet! Call create_visualizer before register it to env")
    return _REGISTERED_VISUALIZER[0].set_sim_instance(sim, model_xml)


def get_visualizer() -> MeshCatVisualizerMujoco:
    if len(_REGISTERED_VISUALIZER) == 0:
        raise RuntimeError(f"No Mujoco Web Visualizer is created.")
    return _REGISTERED_VISUALIZER[0]
