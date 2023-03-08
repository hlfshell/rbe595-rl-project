from math import pi
from random import choice, uniform
from typing import Any, Dict, List, Tuple

import numpy as np
from panda_gym.envs.core import RobotTaskEnv, Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet


class SorterTask(Task):
    def __init__(
        self,
        sim: PyBullet,
        objects_count: int = 5,
        img_size: Tuple[int, int] = (256, 256),
    ):
        super().__init__(sim)

        self.score: float = 0.0

        self.initial_steps: int = 100
        self.steps_remaining = self.initial_steps
        self.steps_sort_extension = 10

        self.objects_count = objects_count
        self.img_size = img_size

        # Create our plane and table for the scenario
        self.sim.create_table(length=1.1, width=0.8, height=0.4, x_offset=-0.3)
        self.sim.create_plane(z_offset=-0.4)

        # Create our sorting areas
        self._init_sorting_areas()

        # goal_positions will contain one of the three
        # preset strings (set below) w/ the resulting
        # starting position each goal is expected to be
        self.sorter_positions: Dict[str, np.array] = {}

        # objects is a dict w/ integer setting (0-4) for
        # the five objects created. The resulting dict
        # contains the "id", "name", initial "position",
        # "shape", and "color" for each object
        self.goal: Dict[int, Dict[str, Any]] = {}

        # Size multiplier is the range of sizes allowed for
        # target objects
        self.size_multiplier: Tuple[float, float] = (0.5, 1.5)

        # These position_limits are where the objects are allowed
        # to spawn. This reads as (x, y), where each axis
        # in turn is a tuple of min/max placement.
        self.position_limits: Tuple[Tuple[float, float]] = ((-0.2, 0.2), (-0.2, 0.2))

    def _init_sorting_areas(self):
        self.sorter_positions = {
            SORTING_ONE: np.array([-0.25, -0.2, 0.01]),
            SORTING_TWO: np.array([-0.25, 0.00, 0.01]),
            SORTING_THREE: np.array([-0.25, 0.2, 0.01]),
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
                orietnation=np.array([0.0, 0.0, 0.0, 1.0]),
            )

    def setup_target_objects(self):
        """
        Generate self.objects_count objects randomly on the table of
        varying sizes, colors, and shapes. The shapes check to ensure
        that they do NOT collide with one another to start.
        """
        base_size = 0.025
        base_mass = 0.5
        base_box_volume = base_size**3
        base_cylinder_volume = pi * base_size**3  # h == r in base cylinder
        # First, delete each object to cleanup
        self.delete_all_objects()

        for object in range(0, self.objects_count):
            # Attempt to create the object. If it collides with
            # another, delete it and try again
            while True:
                name = f"object_{object}"
                color = self.get_random_color()
                shape = choice(SHAPES)
                position = self.get_random_object_position()

                if shape == CUBE:
                    x = base_size * uniform(
                        self.size_multiplier[0], self.size_multiplier[1]
                    )
                    y = base_size * uniform(
                        self.size_multiplier[0], self.size_multiplier[1]
                    )
                    z = base_size * uniform(
                        self.size_multiplier[0], self.size_multiplier[1]
                    )

                    volume = x * y * z
                    mass_multiplier = volume / base_box_volume

                    self.sim.create_box(
                        body_name=name,
                        half_extents=np.array([x, y, z]),
                        mass=base_mass * mass_multiplier,
                        ghost=False,
                        position=position,
                        rgba_color=color,
                    )
                elif shape == CYLINDER:
                    height = base_size * uniform(
                        self.size_multiplier[0], self.size_multiplier[1]
                    )
                    radius = base_size * uniform(
                        self.size_multiplier[0], self.size_multiplier[1]
                    )

                    volume = pi * radius**2 * height
                    mass_multiplier = volume / base_cylinder_volume

                    self.sim.create_cylinder(
                        body_name=name,
                        radius=radius,
                        height=height,
                        mass=base_mass * mass_multiplier,
                        ghost=False,
                        position=position,
                        rgba_color=color,
                    )
                elif shape == SPHERE:
                    multiplier = uniform(
                        self.size_multiplier[0], self.size_multiplier[1]
                    )
                    self.sim.create_sphere(
                        body_name=name,
                        radius=base_size * multiplier,
                        mass=base_mass * multiplier,
                        ghost=False,
                        position=position,
                        rgba_color=color,
                    )
                else:
                    raise Exception("Improper shape chosen")

                id = self.sim._bodies_idx[name]

                # Now ensure that the shape created does not
                # intersect any of the existing shapes
                collisions = False
                # If this is the first, we're good; move on
                if len(self.goal) <= 0:
                    break
                # ...otherwise we're going to compare it
                # against all known objects. If there's
                # overlap we delete this and move on
                for other in self.goal:
                    other_id = self.goal[other]["id"]
                    if self.check_collision(id, other_id):
                        collisions = True
                        break

                if collisions:
                    self.sim.physics_client.removeBody(id)
                    continue
                else:
                    break

            self.goal[object] = {
                "id": id,
                "name": name,
                "position": position,
                "shape": shape,
                "color": color,
            }

    def check_collision(self, object1: str, object2: str) -> bool:
        """
        check_collision will check if the two objects overlap at all
        and returns a boolean to that effect
        """
        contacts = self.sim.physics_client.getContactPoints(object1, object2)
        return contacts is not None and len(contacts) > 0

    def delete_all_objects(self):
        for object in self.goal:
            self.sim.physics_client.removeBody(self.goal[object]["id"])
        self.goal = {}

    def get_random_color(self) -> np.array:
        """
        Returns an appropriate color from a list of decent color choices
        in the form of a 4 dimensional RGBA array (colors are (0,255) ->
        (0, 1) scaled)
        """
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 255),
            (178, 102, 255),
            (102, 255, 255),
            (102, 0, 204),
            (255, 128, 0),
            (204, 0, 102),
        ]
        color = choice(colors)

        return np.array([color[0], color[1], color[2], 0.8])

    def get_random_object_position(self) -> np.array:
        """
        get_random_object_position returns a random np.array of an object's
        permissions within the permissive bounds set at instantiation.
        """
        x = uniform(self.position_limits[0][0], self.position_limits[0][1])
        y = uniform(self.position_limits[1][0], self.position_limits[1][1])
        z = 0.01
        return np.array([x, y, z])

    def reset(self):
        # Ensure each goal hasn't moved
        self.set_sorter_positions()

        # Generate new objects
        self.setup_target_objects()

        self.score = 0
        self.steps_remaining = self.initial_steps

    def get_obs(self) -> np.array:
        """
        get_obs will determine if any objects collided and need to be removed,
        and adjust the score as expected. It will then return an observation,
        which will be the simulation render from the camera angle.
        """
        # Decrement our steps so that we can timeout if this takes
        # too long
        self.steps_remaining -= 1
        # Every step forward gets a penalty for time
        self.score += STEP_PENALTY

        # First check for floor collisions. Remove any colliding objects.
        floor_id = self.sim._bodies_idx["plane"]
        clear_keys: List[str] = []
        for object_key in self.goal:
            object_id = self.goal[object_key]["id"]
            if self.check_collision(object_id, floor_id):
                self.score += FLOOR_PENALTY
                self.sim.physics_client.removeBody(object_id)
                clear_keys.append(object_key)

        # Clear up removed goals
        for key in clear_keys:
            del self.goal[key]

        # Then check for collisions between the goals and a given target.
        # Remove any colliding targets.
        clear_keys: List[str] = []
        for object_key in self.goal:
            for goal in GOALS:
                object = self.goal[object_key]
                object_id = object["id"]
                goal_id = self.sim._bodies_idx[goal]

                if object_key not in clear_keys and self.check_collision(
                    object_id, goal_id
                ):
                    self.sim.physics_client.removeBody(object_id)
                    clear_keys.append(object_key)

                    if CORRECT_SORTS[goal] == object["shape"]:
                        self.score += SORT_REWARD
                    else:
                        self.score += WRONG_SORT_REWARD

                    self.steps_remaining += self.steps_sort_extension

        # Clear up removed goals
        for key in clear_keys:
            del self.goal[key]

        # Ensure that each goal hasn't moved; this is a consequence of the
        # collision checking we do
        self.set_sorter_positions()

        # We have to swap render mode if it's set to human mode
        # to get it to draw for us.
        original_render_mode = self.sim.render_mode
        self.sim.render_mode = "rgb_array"
        img = self.sim.render(
            self.img_size[0],
            self.img_size[1],
            # target_position=self.camera_position,
            target_position=None,
            distance=0.0,
            yaw=45,
            pitch=-30,
            roll=0.0,
        )
        self.sim.render_mode = original_render_mode
        return img

    def get_achieved_goal(self) -> np.ndarray:
        return np.array([len(self.goal) < 0], dtype="bool")

    def is_terminated(self) -> bool:
        """
        is_terminated returns whether or not the episode is
        in a terminal state; this can be due to:
        1. All objects have been removed somehow from the env
        2. The timer has hit 0

        It is not an indication of success
        """
        if self.steps_remaining <= 0:
            return True

        return len(self.goal) <= 0

    def is_success(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any] = ...,
    ) -> np.ndarray:
        """
        is_success is a misnamed function, required as a layover
        from using the panda_gym library. Instead it is best
        to read it as an interface w/ is_terminated, and in no
        way reads whether it was a success, since the episode can
        end via timeout without doing the goals.
        """
        return np.array([self.is_terminated()], dtype="bool")

    def compute_reward(
        self,
        achieved_goal: np.ndarray,
        desired_goal: np.ndarray,
        info: Dict[str, Any] = ...,
    ) -> np.ndarray:
        return np.array([self.score], dtype="float32")


class SorterEnv(RobotTaskEnv):
    """Sorter task wih Panda robot.

    Args:
        render_mode (str, optional): Render mode. Defaults to "human".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        render_width (int, optional): Image width. Defaults to 720.
        render_height (int, optional): Image height. Defaults to 480.
    """

    def __init__(
        self,
        objects_count: int = 5,
        render_mode: str = "human",
        control_type: str = "ee",
        renderer: str = "OpenGL",
        render_width: int = 720,
        render_height: int = 480,
    ) -> None:
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
        task = SorterTask(sim, objects_count=objects_count)
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

CUBE = 0
CYLINDER = 1
SPHERE = 2
SHAPES = [CUBE, CYLINDER, SPHERE]

# This is the expected orrect sorting results
CORRECT_SORTS = {
    SORTING_ONE: CYLINDER,
    SORTING_TWO: SPHERE,
    SORTING_THREE: CUBE,
}

STEP_PENALTY = -1
FLOOR_PENALTY = -50
WRONG_SORT_REWARD = 25
SORT_REWARD = 100
