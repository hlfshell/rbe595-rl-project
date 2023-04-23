from project.ppo.pose_trainer import Trainer
from project.ppo.pose_model import (
    PPOPoseActor,
    PPOPoseCritic,
)

import gymnasium as gym

env = gym.make('Pendulum-v1', render_mode="human")

actor = PPOPoseActor(5, "pendulum")
critic = PPOPoseCritic(5, "pendulum")

trainer = Trainer(
    env,
    actor,
    critic,
    timesteps=200_000_000,
    timesteps_per_batch=500,
    max_timesteps_per_episode=100,
)

trainer.train()