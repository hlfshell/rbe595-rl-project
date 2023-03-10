from project.ppo.model import PPO, PPOImageInput
from panda_gym.envs.core import RobotTaskEnv

from pathlib import Path
from typing import Tuple


class TrainingState:
    def __init__(self):
        pass

    def save(self, filepath: str):
        pass

    def load(self, filepath: str):
        pass


class PPOTrainer:
    def __init__(
        self,
        env: RobotTaskEnv,
        model: PPO,
        image_size: Tuple[int, int],
        images_per_input: int = 4,
        save_folder: str = "./inprogress/pto",
    ):
        self.env = env
        self.model = model
        self.input = PPOImageInput(image_size=image_size, images=images_per_input)

        Path(save_folder).mkdir(parents=True, exist_ok=True)

        self.state = TrainingState()

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
        self.model.load(f"{folder}/models")

    def train(self):
        pass
