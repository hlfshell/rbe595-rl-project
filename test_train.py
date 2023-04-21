from project.envs.sorter import SorterEnv, OBSERVATION_IMAGE, OBSERVATION_POSES
from project.envs.tosser import TosserEnv
from project.ppo.pose_memory import TrainingState
from project.ppo.pose_trainer import Trainer
from project.ppo.pose_model import PPOPoseActor, PPOPoseCritic

import os

env = SorterEnv(OBSERVATION_POSES, 5, render_mode="rgb_array")

actor = PPOPoseActor(5, "ee")
critic = PPOPoseCritic(5, "ee")

# Determine if there is a training state at training/state.data -
# if so, load the TrainingState for it. Otherwise, create a new one.
# Check if training/state.data exists
# if os.exists("./training/state.data"):
#     state = TrainingState.Load("./training/state.data")
# else:
#     state = TrainingState(10_000, 1_000_000, 4800, 5, 5)

trainer = Trainer(
    env,
    actor,
    critic,
    TrainingState(),
)

trainer.train()
