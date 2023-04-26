from panda_gym.envs import PandaPickAndPlaceEnv, PandaReachEnv, PandaStackEnv
from project.ppo.pose_trainer import Trainer
from project.ppo.panda_model import (
    PPOActor,
    PPOCritic,
)

from os import path
from pathlib import Path


# env = PandaPickAndPlaceEnv(render_mode="rgb_array", renderer="OpenGL")
# env = PandaReachEnv(render_mode="rgb_array", renderer="Tiny")
env = PandaStackEnv(render_mode="rgb_array", renderer="Tiny")

input_size = env.observation_space["observation"].shape[0]
output_size = env.action_space.shape[0]

actor = PPOActor(input_size, output_size)
critic = PPOCritic(input_size)

trainer = Trainer(
    env,
    actor,
    critic,
    timesteps=200_000_000,
    timesteps_per_batch=5_000,
    max_timesteps_per_episode=200,
)

# Determine if there is a training state at training/state.data -
# if so, load the TrainingState for it. Otherwise, create a new one.
Path("./training").mkdir(parents=True, exist_ok=True)
if path.isfile("./training/state.data"):
    trainer.load("./training")

trainer.train()
