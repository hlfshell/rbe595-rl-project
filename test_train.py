from project.envs.sorter import SorterEnv, OBSERVATION_IMAGE, OBSERVATION_POSES
from project.envs.tosser import TosserEnv
from project.ppo.pose_memory import TrainingState
from project.ppo.pose_trainer import Trainer
from project.ppo.trainer_with_memory import Trainer as MemoryTrainer
from project.ppo.pose_model import (
    PPOPoseActor,
    PPOPoseCritic2,
    PPOPoseActorMV,
    PPOPoseCritic,
)

import os

env = SorterEnv(OBSERVATION_POSES, 5, render_mode="rgb_array")

# actor = PPOPoseActor(5, "ee")
actor = PPOPoseActorMV(5, "ee")
critic = PPOPoseCritic(5, "ee")
# critic = PPOPoseCritic2(5, "ee")

# Determine if there is a training state at training/state.data -
# if so, load the TrainingState for it. Otherwise, create a new one.
# Check if training/state.data exists
# if os.exists("./training/state.data"):
#     state = TrainingState.Load("./training/state.data")
# else:
#     state = TrainingState(10_000, 1_000_000, 4800, 5, 5)

# trainer = MemoryTrainer(
#     env,
#     actor,
#     critic,
#     TrainingState(10_000, 1_000_000, 4800, 5, 5),
# )

trainer = Trainer(
    env,
    actor,
    critic,
    timesteps=200_000_000,
    timesteps_per_batch=10_000,
    # timesteps_per_batch=1_000,
    max_timesteps_per_episode=250,
)

trainer.train()
