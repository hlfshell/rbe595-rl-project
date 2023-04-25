from project.ppo.pose_trainer import Trainer
from project.ppo.pose_model import (
    PPOPoseActor,
    PPOPoseCritic,
)

import gymnasium as gym

env = gym.make('Pendulum-v1')

actor = PPOPoseActor(5, "pendulum")
critic = PPOPoseCritic(5, "pendulum")

trainer = Trainer(
    env,
    actor,
    critic,
    timesteps=200_000_000,
    timesteps_per_batch=2048,
    max_timesteps_per_episode=200,
)
	# hyperparameters = {
	# 			'timesteps_per_batch': 2048, 
	# 			'max_timesteps_per_episode': 200, 
	# 			'gamma': 0.99, 
	# 			'n_updates_per_iteration': 10,
	# 			'lr': 3e-4, 
	# 			'clip': 0.2,
	# 			'render': True,
	# 			'render_every_i': 10
	# 		  }

trainer.train()