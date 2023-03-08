from math import pi
from random import choice, uniform
from typing import Any, Dict, List, Tuple

import numpy as np
from panda_gym.envs.core import RobotTaskEnv, Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet

from project.envs.sorter import SorterEnv, SorterTask


class TosserTask(SorterTask):

    def __init__(
        self,
        sim: PyBullet,
        objects_count: int = 5,
        img_size: Tuple[int, int] = (256, 256),
    ):
        super().__init__(sim, objects_count=objects_count, img_size=img_size)
        self.object_opacity = 1.0

    def task_init(self):
        # Create our plane and table for the scenario
        self.sim.create_table(length=0.5, width=0.6, height=0.4, x_offset=-0.3)
        self.sim.create_table(length=0.2, width=0.6, height=0.4, x_offset=0.4)

        # These position_limits are where the objects are allowed
        # to spawn. This reads as (x, y), where each axis
        # in turn is a tuple of min/max placement.
        self.object_position_limits: Tuple[Tuple[float, float]] = ((-0.4, -0.1), (-0.2, 0.2))

    def _init_sorting_areas(self):
        self.sorter_positions = {
            SORTING_ONE: np.array([0.4, -0.2, 0.01]),
            SORTING_TWO: np.array([0.4, 0.00, 0.01]),
            SORTING_THREE: np.array([0.4, 0.2, 0.01]),
        }
        self.sim.create_box(
            body_name=SORTING_ONE,
            half_extents=np.array([0.05, 0.1, 0.01]),
            mass=0.0,
            ghost=False,
            position=self.sorter_positions[SORTING_ONE],
            rgba_color=np.array([1.0, 0, 0, 0.4]),
        )
        self.sim.create_box(
            body_name=SORTING_TWO,
            half_extents=np.array([0.05, 0.1, 0.01]),
            mass=0.0,
            ghost=False,
            position=self.sorter_positions[SORTING_TWO],
            rgba_color=np.array([0.0, 1.0, 0, 0.4]),
        )
        self.sim.create_box(
            body_name=SORTING_THREE,
            half_extents=np.array([0.05, 0.1, 0.01]),
            mass=0.0,
            ghost=False,
            position=self.sorter_positions[SORTING_THREE],
            rgba_color=np.array([0, 0, 1.0, 0.4]),
        )

    def set_sorter_positions(self):
        """
        set_goal_positions will ensure that goals are placed
        in the appropriate place in the environment
        """
        for sorter in self.sorter_positions:
            self.sim.set_base_pose(
                sorter,
                position=self.sorter_positions[sorter],
                orientation=np.array([0.0, 0.0, 0.0, 1.0]),
            )


class TosserEnv(RobotTaskEnv):

    def __init__(
        self,
        objects_count: int = 5,
        render_mode: str = "human",
        control_type: str = "ee",
        renderer: str = "OpenGL",
        render_width: int = 720,
        render_height: int = 480,
    ):
        sim = PyBullet(
            render_mode=render_mode,
            background_color=np.array((200, 200, 200)),
            renderer=renderer,
        )
        robot = Panda(
            sim,
            block_gripper=False,
            base_position=np.array([-0.6, 0.0, 0.0]),
            control_type=control_type,
        )
        task = TosserTask(sim, objects_count=objects_count)
        super().__init__(
            robot,
            task,
            render_width=render_width,
            render_height=render_height,
            render_target_position=None,
            render_distance=0.9,
            render_yaw=45,
            render_pitch=-30,
            render_roll=0.0,
        )
        self.sim.place_visualizer(
            target_position=np.zeros(3), distance=0.9, yaw=45, pitch=-30
        )

    def reset(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        with self.sim.no_rendering():
            self.robot.reset()
            self.task.reset()
        observation = self._get_obs()
        return observation, None

    def _get_obs(self) -> Dict[str, np.ndarray]:
        observation = self.task.get_obs().astype(np.float32)
        achieved_goal = self.task.get_achieved_goal().astype(np.float32)
        return {
            "observation": observation,
            "achieved_goal": achieved_goal,
        }
    


SORTING_ONE = "sorting_one"
SORTING_TWO = "sorting_two"
SORTING_THREE = "sorting_three"
GOALS = [SORTING_ONE, SORTING_TWO, SORTING_THREE]