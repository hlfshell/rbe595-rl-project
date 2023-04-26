import math
from math import pi
from random import choice, uniform
from typing import Any, Dict, Tuple

import numpy as np
from panda_gym.envs.core import RobotTaskEnv, Task
from panda_gym.envs.robots.panda import Panda
from panda_gym.pybullet import PyBullet


class TargetObject:
    """
    TargetObject tracks the lifecycle of a target object (not a goal)
    """

    def __init__(
        self,
        id: str,
        name: str,
        shape: int,
        size: np.array,
        position: np.array,
        color: np.array,
        removed: bool = False,
    ):
        self.id = id
        self.name = name
        self.shape = shape
        self.size = size
        self.position = position
        self.color = color
        self.removed = removed


class SorterTask(Task):
    def __init__(
        self,
        sim: PyBullet,
        observation_type: int,
        robot: Panda,
        objects_count: int = 5,
        img_size: Tuple[int, int] = (256, 256),
        blocker_bar: bool = True,
    ):
        if observation_type not in [OBSERVATION_POSES, OBSERVATION_IMAGE]:
            raise Exception(
                f"Invalid output type {observation_type}. Must be one of "
                + f"{OBSERVATION_POSES} for values and {OBSERVATION_IMAGE} "
                + "four images."
            )

        super().__init__(sim)

        self.robot = robot
        self.observation_type = observation_type

        self.score: float = 0.0

        self.objects_count: int = objects_count
        if observation_type == OBSERVATION_IMAGE:
            self.img_size = img_size
        self.object_opacity = 0.8

        self.sim.create_plane(z_offset=-0.4)

        # goal_positions will contain one of the three
        # preset strings (set below) w/ the resulting
        # starting position each goal is expected to be
        self.sorter_positions: Dict[str, np.array] = {}
        self.blocker_bar = blocker_bar
        self._init_sorting_areas()

        # Track each target object as part of our goal
        self.goal: Dict[int, TargetObject] = {}

        # Size multiplier is the range of sizes allowed for
        # target objects
        self.size_multiplier: Tuple[float, float] = (0.5, 1.5)

        self.task_init()

    def task_init(self):
        # Create our plane and table for the scenario
        self.sim.create_table(length=0.8, width=0.8, height=0.4, x_offset=-0.3)

        # These position_limits are where the objects are allowed
        # to spawn. This reads as (x, y), where each axis
        # in turn is a tuple of min/max placement.
        self.object_position_limits: Tuple[Tuple[float, float]] = (
            (-0.06, 0.06),
            (-0.2, 0.2),
        )
        # self.sim.create_sphere(
        #     body_name="marker",
        #     radius=0.025,
        #     mass=0.0,
        #     ghost=True,
        #     position=(0.06, 0.2, 0.01),
        #     rgba_color=(255, 255, 0, 1.0),
        # )

    def _init_sorting_areas(self):
        self.sorter_positions = {
            SORTING_ONE: np.array([-0.25, -0.2, 0.01]),
            SORTING_TWO: np.array([-0.25, 0.00, 0.01]),
            SORTING_THREE: np.array([-0.25, 0.2, 0.01]),
        }
        if self.blocker_bar:
            self.sorter_positions["blocker"] = np.array([-0.2, 0.0, 0.01])

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
            rgba_color=np.array([0, 0, 1.0, 0.5]),
        )
        if self.blocker_bar:
            # Create the blocking bar
            self.sim.create_box(
                body_name="blocker",
                half_extents=np.array([0.01, 0.3, 0.005]),
                mass=0.0,
                ghost=False,
                position=self.sorter_positions["blocker"],
                rgba_color=np.array([0.0, 0.0, 0.0, 0.8]),
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
                    size = np.array([x, y, z]).astype(np.float32)

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
                    size = np.array([height, radius, 0.0]).astype(np.float32)

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
                    size = np.array([multiplier, 0.0, 0.0]).astype(np.float32)

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
                    other_id = self.goal[other].id
                    if self.check_collision(id, other_id):
                        collisions = True
                        break

                if collisions:
                    self.sim.physics_client.removeBody(id)
                    continue
                else:
                    break

            self.goal[object] = TargetObject(id, name, shape, size, position, color)

    def check_collision(self, object1: str, object2: str) -> bool:
        """
        check_collision will check if the two objects overlap at all
        and returns a boolean to that effect
        """
        contacts = self.sim.physics_client.getContactPoints(object1, object2)
        return contacts is not None and len(contacts) > 0

    def delete_all_objects(self):
        for object in self.goal:
            self.sim.physics_client.removeBody(self.goal[object].id)
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

        return np.array([color[0], color[1], color[2], self.object_opacity])

    def get_random_object_position(self) -> np.array:
        """
        get_random_object_position returns a random np.array of an object's
        permissions within the permissive bounds set at instantiation.
        """
        x = uniform(
            self.object_position_limits[0][0], self.object_position_limits[0][1]
        )
        y = uniform(
            self.object_position_limits[1][0], self.object_position_limits[1][1]
        )
        z = 0.01
        return np.array([x, y, z])

    def reset(self):
        # Ensure each goal hasn't moved
        self.set_sorter_positions()

        # Generate new objects
        self.setup_target_objects()

        # Clear our score
        self.score = 0.0

    def get_obs(self) -> np.array:
        """
        get_obs will determine if any objects collided and need to be removed,
        and adjust the score as expected. It will then return an observation,
        which, depending on the observation type, will be:
            OBSERVATION_POSES: will be a series of values. See _get_obs_poses
                for an explanation of this output.
            OBSERVATION_IMAGE: will be the simulation render from the camera
                angle.
        """
        # First check for floor collisions. Remove any colliding objects.
        floor_id = self.sim._bodies_idx["plane"]
        for object_key in self.goal:
            if self.goal[object_key].removed:
                continue
            object_id = self.goal[object_key].id
            if self.check_collision(object_id, floor_id):
                self.score += FLOOR_PENALTY
                self.sim.physics_client.removeBody(object_id)
                self.goal[object_key].removed = True

        # Then check for collisions between the goals and a given target.
        # Remove any colliding targets.
        for object_key in self.goal:
            if self.goal[object_key].removed:
                continue
            for goal in GOALS:
                object = self.goal[object_key]
                object_id = object.id
                goal_id = self.sim._bodies_idx[goal]

                if self.check_collision(object_id, goal_id):
                    self.sim.physics_client.removeBody(object_id)
                    self.goal[object_key].removed = True

                    if CORRECT_SORTS[goal] == object.shape:
                        self.score += SORT_REWARD
                    else:
                        self.score += WRONG_SORT_REWARD

        # Ensure that each goal hasn't moved; this is a consequence of the
        # collision checking we do
        self.set_sorter_positions()

        if self.observation_type == OBSERVATION_IMAGE:
            return self._get_img()
        else:
            return self._get_poses_output()

    def get_object_pose(self, object: TargetObject) -> np.array:
        object_position = self.sim.get_base_position(object.name)
        object_rotation = self.sim.get_base_rotation(object.name)
        object_velocity = self.sim.get_base_velocity(object.name)
        object_angular_velocity = self.sim.get_base_angular_velocity(object.name)
        observation = np.concatenate(
            [object_position, object_rotation, object_velocity, object_angular_velocity]
        )
        return observation.astype(np.float32)

    def _get_poses_output(self) -> np.array:
        """
        _get_poses_output will return the poses of all objects in the scene,
        as well as their identity and size. It will be a series of values for
        the raw (x, y, z, theta, phi, psi) pose of the object, as well as an
        identity (type of shape), and size (0-1) for min/max size, and the pose
        of the robot's end effector, and a 0-1 value for its gripper open/close
        state. An example of the return would be:

        [[x, y, z, theta, phi, psi,
          xd, yd, zd, thetad, phid, psid, <~ velocities
        [identity], [size]] (times # of set objects), ...,
        (ee_x, ee_y, ee_z, ee_theta, ee_phi, ee_psi,
        ee_xd, ee_yd, ee_zd, ee_thetad, ee_phid, ee_psid), # <~ velocities
        gripper_status]

        Note that the identity is a one-hot encoded list of shape [CUBE, CYLINDER,
        SPHERE] and that size is a three value array of varying meaning based
        on size: [x, y, z] for CUBE, [radius, height] for CYLINDER, [radius]
        for SPHERE. Unused values are 0.0.
        """
        # The size of this vector is determined by the number of objects expected
        pose_values = 12
        shape_values = 3
        size_values = 3
        # End effector values
        ee_values = 12
        finger_values = 1
        # The length of our vector is (pose_values + (identity, size)) for each
        # object, pose_values for the end effector, and one additional value for
        # the gripper finger state (distance between fingers)
        size = (
            (len(self.goal) * (pose_values + shape_values + size_values))
            + ee_values
            + finger_values
        )
        observation: np.array = np.zeros((size,), dtype="float32")

        index = 0
        for object in self.goal.values():
            # If the object has been removed, just report 0's for its existence
            if object.removed:
                index += 1
                continue

            pose = self.get_object_pose(object)
            object_index = index * (pose_values + shape_values + size_values)
            observation[object_index : object_index + pose_values] = pose

            # The shape is a one hot encoded vector of [CUBE, CYLINDER, SPHERE]
            if object.shape == CUBE:
                shape_type = [1, 0, 0]
            elif object.shape == CYLINDER:
                shape_type = [0, 1, 0]
            elif object.shape == SPHERE:
                shape_type = [0, 0, 1]
            shapes_index = object_index + pose_values
            observation[shapes_index : shapes_index + shape_values] = shape_type
            size_index = shapes_index + shape_values
            observation[size_index : size_index + size_values] = object.size
            index += 1

        # Get the end effector position
        ee_position = self.robot.get_ee_position()
        ee_rotation_quaternion = self.sim.get_link_orientation(
            self.robot.body_name, self.robot.ee_link
        )
        ee_rotation = self._quaternion_to_euler(ee_rotation_quaternion)
        # print("rot", ee_rotation)
        # print("rot other", self.sim.get_base_rotation(self.robot.ee_link))
        ee_velocity = self.robot.get_ee_velocity()
        ee_rotational_velocity = self.sim.get_link_angular_velocity(
            self.robot.body_name, self.robot.ee_link
        )

        # ee_angulary_velocity = 0.0
        fingers_width = self.robot.get_fingers_width()
        ee_index = (pose_values + shape_values + size_values) * len(self.goal)
        observation[ee_index : ee_index + 3] = ee_position
        observation[ee_index + 3 : ee_index + 6] = ee_rotation
        observation[ee_index + 6 : ee_index + 9] = ee_velocity
        observation[ee_index + 9 : ee_index + 12] = ee_rotational_velocity
        observation[ee_index + 12] = fingers_width

        return observation

    def _quaternion_to_euler(self, quaternion: np.array):
        """
        _quaternion_to_euler will convert a quaternion to euler angless
        """
        x, y, z, w = quaternion
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        X = math.atan2(t0, t1)

        t2 = 2.0 * (w * y - z * x)
        t2 = 1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.asin(t2)

        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        Z = math.atan2(t3, t4)

        return np.array([X, Y, Z]).astype(np.float32)

    def _get_img(self) -> np.array:
        """
        _get_img will return the image from the camera in human rendering
        mode
        """
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
        return np.array(
            all(target.removed for target in self.goal.values()), dtype="bool"
        )

    def is_terminated(self) -> bool:
        """
        is_terminated returns whether or not the episode is
        in a terminal state; this can be due to:
        1. All objects have been removed somehow from the env
        2. The timer has hit 0

        It is not an indication of success
        """

        return all(obj.removed for obj in self.goal.values())

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
        observation_type: int,
        objects_count: int = 5,
        blocker_bar: bool = True,
        render_mode: str = "human",
        control_type: str = "ee",
        renderer: str = "OpenGL",
        render_width: int = 720,
        render_height: int = 480,
    ) -> None:
        if observation_type not in [OBSERVATION_IMAGE, OBSERVATION_POSES]:
            raise ValueError("observation_type must be one of either images or poses")

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
        task = SorterTask(sim, observation_type, robot, objects_count=objects_count, blocker_bar=blocker_bar)
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

    def get_obs(self) -> np.ndarray:
        return self.task.get_obs().astype(np.float32)

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        score_prior = self.task.score
        self.robot.set_action(action)
        self.sim.step()
        observation = self._get_obs()
        score_after = self.task.score

        # An episode is terminated iff the agent has reached the target
        terminated = bool(
            self.task.is_success(observation["achieved_goal"], self.task.get_goal())
        )
        truncated = False
        info = {"is_success": terminated}
        step_penalty = STEP_PENALTY
        reward = (score_after - score_prior) + step_penalty
        return observation, reward, terminated, truncated, info


SORTING_ONE = "sorting_one"
SORTING_TWO = "sorting_two"
SORTING_THREE = "sorting_three"
GOALS = [SORTING_ONE, SORTING_TWO, SORTING_THREE]

CUBE = 0
CYLINDER = 1
SPHERE = 2
SHAPES = [CUBE, CYLINDER, SPHERE]

# This is the expected correct sorting results
CORRECT_SORTS = {
    SORTING_ONE: CYLINDER,
    SORTING_TWO: SPHERE,
    SORTING_THREE: CUBE,
}

STEP_PENALTY = -1
FLOOR_PENALTY = -50
# WRONG_SORT_REWARD = 25
# SORT_REWARD = 100
WRONG_SORT_REWARD = 200
SORT_REWARD = 500

OBSERVATION_POSES: int = 0
OBSERVATION_IMAGE: int = 1
