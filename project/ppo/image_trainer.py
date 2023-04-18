from project.ppo.image_model import PPOImageActor, PPOImageCritic, PPOImageInput
from panda_gym.envs.core import RobotTaskEnv
import numpy as np

from pathlib import Path
from typing import Tuple, List
from collections import deque


class TrainingState:
    def __init__(
        self,
        timesteps: int,
        image_size: Tuple[int, int],
        images_per_input: int,
        max_observation_memory: int,
        timesteps_per_batch: int,
    ):
        self.timesteps_total = timesteps
        self.timestep: int = 0
        self.timesteps_per_batch = timesteps_per_batch

        self.image_size = image_size
        self.images_per_input = images_per_input

        self.observations: List[np.array] = deque(maxlen=max_observation_memory)
        self.current_observation = PPOImageInput(maxlen=images_per_input)

        self.batch_actions: List[np.array] = deque(maxlen=max_observation_memory)

        # batch_obs = []             # batch observations
        # batch_acts = []            # batch actions
        # batch_log_probs = []       # log probs of each action
        # batch_rews = []            # batch rewards
        # batch_rtgs = []            # batch rewards-to-go
        # batch_lens = []            # episodic lengths in batch

    def get_observation(self, index: int) -> PPOImageInput:
        """
        get_observation takes in an index and builds a PPOImageInput
        from the given spot in time. We do this to avoid saving the
        same images repeatedly.
        """
        if index >= len(self.observations):
            raise Exception(
                f"Index {index} does not exist for "
                + "observations of length {len(self.observations)}"
            )

        observation = PPOImageInput(
            image_size=self.image_size, images=self.images_per_input
        )
        for i in range(index - self.images_per_input + 1, index + 1):
            if i < 0:
                observation.append(np.zeros(self.image_size + (3,)))
            else:
                observation.append(self.observations[i])

        return observation

    def save(self, filepath: str):
        pass

    def load(self, filepath: str):
        pass


class PPOTrainer:
    def __init__(
        self,
        env: RobotTaskEnv,
        actor: PPOActor,
        critic: PPOCritic,
        timesteps: int,
        image_size: Tuple[int, int],
        images_per_input: int = 4,
        timesteps_per_batch: int = 4800,
        max_observation_memory: int = 1_000_000,
        save_folder: str = "./inprogress/pto",
    ):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.input = PPOImageInput(image_size=image_size, images=images_per_input)

        Path(save_folder).mkdir(parents=True, exist_ok=True)

        self.state = TrainingState(
            timesteps,
            image_size,
            timesteps_per_batch,
            images_per_input=images_per_input,
            max_observation_memory=max_observation_memory,
        )

        if self.should_reload:
            self.load_trainer_state()

    def should_reload(self) -> bool:
        """
        should_reload checks to see if there exists checkpoints to load from;
        if so, it will return True
        """
        pass

    def load_trainer_state(self):
        """
        load_trainer_state will reload all necessary checkpoints and memory training
        from the last checkpoint possible. This is not just the model weights,
        but also state memories and training loss progress, etc.
        """

        # Determine the most recent checkpoint by foldername

        folder = ""

        # Load each element into our trainer
        self.state.load(f"{folder}/state/state.pkl")
        self.actor.load(f"{folder}/models/actor.pth")
        self.critic.loadf(f"{folder}/models/critic.pth")

    def train(self):

        tpb = 0  # timesteps per this batch
        while tpb < self.state.timesteps_per_batch:
            tpb += 1

            rewards

        obs, _ = self.env.reset()
        done = False

        for ep_t in range(self.max_timesteps_per_episode):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            if done:
                break
