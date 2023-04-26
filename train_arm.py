from project.envs.sorter import SorterEnv, OBSERVATION_POSES
from project.ppo.pose_trainer import Trainer
from project.ppo.model import (
    Actor,
    Critic,
)

from os import path
from pathlib import Path

env = SorterEnv(
    OBSERVATION_POSES,
    3,
    # render_mode="human",
    # renderer="OpenGL",
    render_mode="rgb_array",
    renderer="Tiny",
    blocker_bar=True,
)

actor = Actor(env)
critic = Critic(env)

trainer = Trainer(
    env,
    actor,
    critic,
    timesteps=200_000_000,
    timesteps_per_batch=5_000,
    max_timesteps_per_episode=750,
)

# Determine if there is a training state at training/state.data -
# if so, load the TrainingState for it. Otherwise, create a new one.
Path("./training").mkdir(parents=True, exist_ok=True)
if path.isfile("./training/state.data"):
    trainer.load("./training")

trainer.train()
