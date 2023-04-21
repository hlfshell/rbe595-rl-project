from __future__ import annotations
from typing import Tuple, List
from random import randint
from torch import Tensor

import torch
import pickle
import os

import numpy as np

"""
EpisodeMemory is a tuple of the following:
    - observations: List[np.array]
    - actions: List[np.array]
    - log_probabilities: List[np.array]
    - rewards: List[float]

    Note that the rewards is the calculated discounted
    rewards, not the raw rewards.
"""
EpisodeMemory: Tuple[List[np.array], List[np.array], List[np.array], List[float]]


class TrainingState:
    def __init__(
        self,
        γ: float = 0.99,
        λ: float = 0.95,
        ε: float = 0.2,
        α: float = 0.005,
        epochs: int = 10_000,
        max_observation_memory: int = 250_000,
        episode_timestep_max: int = 200,
        timesteps_per_batch: int = 5_000,
        save_every_epochs: int = 5,
        training_cycles_per_batch: int = 1,
    ):
        # ================
        # State management parameters
        # ================

        # max_memory represents the maximum number
        # of timesteps we'll store during training.
        # Once this is exceeded additional cleanup
        # functionality will be handled
        self.max_memory = max_observation_memory

        # training_cycles_per_batch is the number of
        # training cycles we'll perform per batch
        self.training_cycles_per_batch = training_cycles_per_batch

        # ================
        # Trainer state
        # ================

        # timesteps is the current timesteps experienced
        # across all of our rollouts
        self.timesteps: int = 0

        # epochs is the total number of
        self.epochs: int = epochs

        # epochs_completed is the number of epochs we've
        # completed
        self.epochs_completed: int = 0

        # How many timesteps do we load into each batch
        self.timesteps_per_batch = timesteps_per_batch

        # How long will we let each episode run before terminating?
        self.episode_timestep_max = episode_timestep_max

        # rewards_all tracks all end of episode rewards for all
        # time, outside of our batch memory, for tracking progress
        # of the model throughout training
        self.rewards_all: List[float] = []

        # save_every_epochs is the number of epochs we wait
        # before saving the current state of the model
        self.save_every_epochs = save_every_epochs

        # ================
        # Trainer hyperparameters
        # ================

        # λ our smoothing constant
        self.λ = 0.95

        # γ is our discount factor for calculating rewards
        self.γ = 0.99

        # ε is our clipping factor for the PPO loss function
        self.ε = 0.2

        # learning_rate is the learning rate for our optimizer
        self.α = 0.005

        # ================
        # Batch Memory
        # ================

        # observations are a list of states derived from
        # our environment
        self.observations: List[np.array] = []

        # These are actions performed by our agent at
        # the given timestep during exploration
        self.actions: List[np.array] = []

        # We need to track the log probabilities of each action
        # for the given distribution at that time.
        self.log_probabilities: List[np.array] = []

        # rewards tracks the rewards of the given memory. Note
        # that these rewards are discounted across the episode
        # already
        self.rewards: List[float] = []

        # episode_lengths is key to our memory system. We
        # wish to grab chunks in set episodes for similar
        # behavior during training. This allows us to remove
        # all data and retrieve in chunks of per-episode
        # despite our simple appending of other data.
        self.episode_lengths: List[int] = []

    def remember(
        self,
        observations: List[np.array],
        actions: List[np.array],
        log_probabilities: List[np.array],
        rewards: List[float],
    ):
        """
        remember takes the output of a given episode and places it
        into our trianing memory. If necessary, it handles unloading
        the oldest episodes to make room
        """
        episode_timesteps = len(observations)

        # Add the observations, actions, log probabilities, and rewards
        # to our memory
        self.observations.extend(observations)
        self.actions.extend(actions)
        self.log_probabilities.extend(log_probabilities)
        self.rewards.extend(rewards)
        self.rewards_all.append(rewards[-1])

        # Add the episode length to our episode lengths
        self.episode_lengths.append(episode_timesteps)

        # Update the total timesteps
        self.timesteps += episode_timesteps

        # Ensure that our memory limit isn't surpassed
        while len(self.observations) > self.max_memory:
            self.unload_oldest_episode()

    def unload_oldest_episode(self):
        """
        unload_oldest_episode removes the oldest episode from our memory
        """
        # Get the length of the oldest episode
        oldest_episode_length = self.episode_lengths.pop(0)

        # Remove the oldest episode's data from our memory
        self.observations = self.observations[oldest_episode_length:]
        self.actions = self.actions[oldest_episode_length:]
        self.log_probabilities = self.log_probabilities[oldest_episode_length:]
        self.rewards = self.rewards[oldest_episode_length:]

    def get_episode(self, index: int) -> EpisodeMemory:
        """
        get_episode retrieves the episode at the given index
        """
        episode_length = self.episode_lengths[index]
        # Get the start and end of the episode
        start = sum(self.episode_lengths[:index])
        end = start + episode_length

        # Return the episode's data
        return (
            self.observations[start:end],
            self.actions[start:end],
            self.log_probabilities[start:end],
            self.rewards[start:end],
        )

    def get_batch(
        self, offset: int = 0
    ) -> Tuple[np.array, np.array, np.array, np.array]:
        """
        get_batch retrieves a batch of data from our memory
        preserving grouping of episodes until the cutoff batch size
        """
        # observations, _, log_probabilities, rewards
        observations: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        log_probabilities: List[np.ndarray] = []
        rewards: List[np.ndarray] = []

        batch_timesteps = 0
        episode_index = offset
        while batch_timesteps < self.timesteps_per_batch:
            # If we are requesting a batch beyond what
            # we have, then it's time to just end
            if episode_index >= len(self.episode_lengths):
                break

            # Get the episode
            obs, acts, log_probs, rwds = self.get_episode(episode_index)

            # Add the episode to the batches
            observations.append(obs)
            actions.append(acts)
            log_probabilities.append(log_probs)
            rewards.append(rwds)

            # The timesteps in the episode is the length of
            # any of its datums
            batch_timesteps += len(obs)

            episode_index += 1

        # Trim the batch if necessary - due to the way we
        # compiled the batch we should only need to trim the
        # last episode
        if batch_timesteps > self.timesteps_per_batch:
            # Get the difference between the batch timesteps
            # and the desired timesteps
            difference = batch_timesteps - self.timesteps_per_batch

            # For each of our batched datums we remove a number of
            # timesteps equal to our difference
            observations = observations[:-difference]
            actions = actions[:-difference]
            log_probabilities = log_probabilities[:-difference]
            rewards = rewards[:-difference]

        return observations, actions, log_probabilities, rewards

    def memory_to_tensor(
        self, memory: EpisodeMemory
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        memory_to_tensor converts a given episode memory to a set of
        tensors
        """
        # For a speed optimization, ensure that we convert the data
        # to a numpy array prior to converting to tensor.
        observations_array = np.array(memory[0], dtype=np.float32)
        actions_array = np.array(memory[1], dtype=np.float32)
        log_probabilities_array = np.array(memory[2], dtype=np.float32)
        rewards_array = np.array(memory[3], dtype=np.float32)

        # Convert the episode memory to a tensor
        observations = Tensor(observations_array, dtype=torch.float32)
        actions = Tensor(actions_array, dtype=torch.float32)
        log_probabilities = Tensor.tensor(log_probabilities_array, dtype=torch.float32)
        rewards = Tensor(rewards_array, dtype=torch.float32)

        return observations, actions, log_probabilities, rewards

    def batch_to_tensor(
        self, batch: List[EpisodeMemory]
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        batch_to_tensor converts a given batch to a set of tensors
        for training
        """
        # Convert the batch to a tensor
        observations = Tensor(batch[0], dtype=torch.float32)
        actions = Tensor(batch[1], dtype=torch.float32)
        log_probabilities = Tensor(batch[2], dtype=torch.float32)
        rewards = Tensor(batch[3], dtype=torch.float32)

        return observations, actions, log_probabilities, rewards

    def save(self, path: str):
        """
        save saves the current state of the training
        """

        # First we create a dict of all possible data to save
        data = {
            "observations": self.observations,
            "actions": self.actions,
            "log_probabilities": self.log_probabilities,
            "rewards": self.rewards,
            "episode_lengths": self.episode_lengths,
            "rewards_all": self.rewards_all,
            "timesteps": self.timesteps,
            "max_memory": self.max_memory,
            "save_every_epochs": self.save_every_epochs,
            "episode_timestep_max": self.episode_timestep_max,
            "timesteps_per_batch": self.timesteps_per_batch,
            "training_cycles_per_batch": self.training_cycles_per_batch,
            "epochs": self.epochs,
            "epochs_completed": self.epochs_completed,
            "γ": self.γ,
            "λ": self.λ,
            "ε": self.ε,
            "α": self.α,
        }

        # Ensure the path exists with appropriate subdirectories
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def Load(path: str) -> TrainingState:
        """
        Load loads a training state from a file and returns
        the resulting training state
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        # Create a new training state
        training_state = TrainingState(
            data["γ"],
            data["λ"],
            data["ε"],
            data["α"],
            data["epochs"],
            data["max_memory"],
            data["episode_timestep_max"],
            data["timesteps_per_batch"],
            data["save_every_epochs"],
            data["training_cycles_per_batch"],
        )

        # Load the data from the file
        training_state.observations = data["observations"]
        training_state.actions = data["actions"]
        training_state.log_probabilities = data["log_probabilities"]
        training_state.rewards = data["rewards"]
        training_state.episode_lengths = data["episode_lengths"]
        training_state.rewards_all = data["rewards_all"]
        training_state.timesteps = data["timesteps"]
        training_state.epochs_completed = data["epochs_completed"]

        return training_state
