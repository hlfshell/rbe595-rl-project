from __future__ import annotations
from project.ppo.pose_model import PPOPoseActor, PPOPoseCritic
from panda_gym.envs.core import RobotTaskEnv
from torch.distributions import Normal
from torch import Tensor
from torch.nn import MSELoss
import numpy as np
from project.ppo.pose_memory import TrainingState

import torch
from pathlib import Path
from typing import Tuple, List
from collections import deque

EpisodeMemory: Tuple[List[np.array], List[np.array], List[np.array], List[float]]


class Trainer:
    def __init__(
        self,
        env: RobotTaskEnv,
        actor: PPOPoseActor,
        critic: PPOPoseCritic,
        state: TrainingState,
    ):
        self.env = env
        self.actor = actor
        self.critic = critic
        self.state = state

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.state.α
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.state.α
        )

        # Initialize some status tracking memory
        self.previous_print_length: int = 0
        self.current_action = "Initializing"

    def print_status(self):
        latest_reward = 0.0
        average_reward = 0.0
        avg_episode_length = 0.0

        if len(self.state.rewards_all) > 0:
            latest_reward = self.state.rewards_all[-1]
            average_reward = np.mean(self.state.rewards_all[-100:])
            avg_episode_length = np.mean(self.state.episode_lengths[-100:])

        # msg = (
        #     f"Epoch {self.state.epochs_completed + 1} | {self.current_action} | "
        #     + f"Latest Reward: {round(latest_reward)} | Latest Avg Rewards: {round(average_reward, 2)} | "
        #     + f"Total Timesteps: {self.state.timesteps:,} | Avg Episode Length: {avg_episode_length}"
        # )

        # print(" " * self.previous_print_length, end="\r")
        # print(msg, end="\r")
        msg = f'''
            =========================================
            Epoch {self.state.epochs_completed + 1}
            {self.current_action}
            Latest Reward: {round(latest_reward)}
            Latest Avg Rewards: {round(average_reward, 2)}
            Total Timesteps: {self.state.timesteps:,}
            Avg Episode Length: {avg_episode_length}
            =========================================
        '''

        print(msg)
        # self.previous_print_length = len(msg)

    def save(self, directory: str):
        """
        save will save the models, state, and any additional
        data to the given directory
        """
        self.actor.save(f"{directory}/actor.pth")
        self.critic.save(f"{directory}/critic.pth")
        self.state.save(f"{directory}/state.data")

    @staticmethod
    def Load(directory: str) -> Trainer:
        """
        Load will load the models, state, and any additional
        data from the given directory
        """
        print(f"... Loading models and state from {directory} ...")
        actor = PPOPoseActor.Load(f"{directory}/actor.pth")
        critic = PPOPoseCritic.Load(f"{directory}/critic.pth")
        state = TrainingState.Load(f"{directory}/state.data")

        return Trainer(state.env, actor, critic, state)

    def initial_rollout(self):
        """
        initial_rollout will perform episode runs of the environment
        until we have an initial minimum number of timesteps per
        our training state requirements.
        """
        while self.state.timesteps < self.state.timesteps_per_batch:
            self.rollout()

    def rollout(self) -> EpisodeMemory:
        """
        rollout will perform a rollout of the environment and
        return the memory of the episode with the current
        actor model
        """
        self.env.reset()

        timesteps = 0
        observations: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        log_probabilities: List[float] = []
        rewards: List[float] = []

        while True:
            timesteps += 1

            observation = self.env.get_obs()
            action_distribution = self.actor(observation)
            action = action_distribution.sample()
            log_probability = action_distribution.log_prob(action).detach().numpy()
            action = action.detach().numpy()
            observation, reward, terminated, _, _ = self.env.step(action)
            observation = observation["observation"]

            observations.append(observation)
            actions.append(action)
            log_probabilities.append(log_probability)
            rewards.append(reward)

            if timesteps >= self.state.episode_timestep_max:
                terminated = True

            if terminated:
                # Penalize for each timestep
                rewards[-1] -= timesteps
                break

        # Calculate the discounted rewards for this episode
        discounted_rewards = self.calculate_discounted_rewards(rewards)

        self.state.remember(
            observations, actions, log_probabilities, discounted_rewards
        )

        return observations, actions, log_probabilities, discounted_rewards

    def calculate_discounted_rewards(self, rewards: List[float]) -> List[float]:
        """
        calculated_discounted_rewards will calculate the discounted rewards
        of each timestep of an episode given its initial rewards and episode
        length
        """
        discounted_rewards: List[float] = []
        discounted_reward: float = 0
        for reward in reversed(rewards):
            discounted_reward = reward + (self.state.γ * discounted_reward)
            # We insert here to append to the front as we're calculating
            # backwards from the end of our episodes
            discounted_rewards.insert(0, discounted_reward)

        return discounted_rewards

    def training_step(self) -> Tuple[float, float]:
        """
        training_step will perform a single epoch of training for the
        actor and critic model

        Returns the loss for each model at the end of the step
        """
        # This will track which episodes we've trained on; we will
        # train on our entire memory for each epoch
        episodes_offset = 0

        training_cycles = 0

        while True:
            # Pull a batch of data from our memory
            observations, _, log_probabilities, rewards = self.state.get_batch(
                episodes_offset
            )

            # Increment our episodes offset by the number of episodes we've
            # pulled
            episodes_offset += len(observations)

            # If our observations is empty, then we have trained on all
            # available data and can return
            if len(observations) == 0:
                training_cycles += 1
                if training_cycles >= self.state.training_cycles_per_batch:
                    break
                else:
                    continue

            # For our given batch we need to get the current estimated
            # value of our given states for our critic, V
            V = self.critic(observations).detach().squeeze()

            # Now we need to calculate our advantage and normalize it
            advantage = Tensor(np.array(rewards, dtype="float32")) - V
            normalized_advantage = (advantage - advantage.mean()) / (
                advantage.std() + 1e-8
            )

            # Get our output for the current actor given our log
            # probabilities
            current_action_distributions = self.actor(observations)
            current_actions = current_action_distributions.sample()
            current_log_probabilities = current_action_distributions.log_prob(
                current_actions
            )

            # We are calculating the ratio as defined by:
            #
            #   π_θ(a_t | s_t)
            #   --------------
            #   π_θ_k(a_t | s_t)
            # Where our originaly utilized log probabilities are
            # π_θ_k and our current model is creating π_θ. We
            # use the log probabilities and subtract, then raise
            # e to the power of the results to simplify the math
            # for back propagation/gradient descent.
            # Note that we have a log probability matrix of shape
            # (batch size, number of actions), where we're expecting
            # (batch size, 1). We sum our logs as the log(A + B) =
            # log(A) + log(B).
            log_probabilities = Tensor(np.array(log_probabilities, dtype=np.float32))
            log_probabilities = torch.sum(log_probabilities, dim=-1)
            current_log_probabilities = torch.sum(current_log_probabilities, dim=-1)
            ratio = torch.exp(current_log_probabilities - log_probabilities)

            # Now we calculate the actor loss for this step
            actor_loss = -torch.min(
                ratio * normalized_advantage,
                torch.clamp(ratio, 1 - self.state.ε, 1 + self.state.ε)
                * normalized_advantage,
            ).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Now we do a training step for the critic

            # Calculate what the critic current evaluates our states as.
            # First we have the critic evaluate all observation states,
            # then compare it ot the collected rewards over that time.
            # We will convert our rewards into a known tensor
            V = self.critic(observations)
            reward_tensor = Tensor(rewards).unsqueeze(-1)
            critic_loss = MSELoss()(V, reward_tensor)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        return

    def train(self):
        """
        train will train the actor and critic models with the
        given training state
        """
        # If our training is starting and we don't have enough
        # episodes for our batch, initialize by performing
        # multiple epsiodes
        if self.state.timesteps < self.state.timesteps_per_batch:
            self.current_action = "Initial Rollout"
            self.print_status()
            self.initial_rollout()

        while self.state.epochs_completed <= self.state.epochs:
            # Perform enough rollouts to fill a new batch prior
            # to requesting a new one
            self.print_status()
            new_timesteps = 0
            while new_timesteps < self.state.timesteps_per_batch:
                self.current_action = "Rollout"
                states, _, _, _ = self.rollout()
                self.print_status()
                new_timesteps += len(states)

            # Now that we've performed sufficient exploration, run
            # a certain amount of taining steps
            self.current_action = "Training"
            self.print_status()
            self.training_step()

            if self.state.epochs_completed + 1 % self.state.save_every_epochs:
                self.current_action = "Saving"
                self.print_status()
                self.save("training")

            self.state.epochs_completed += 1

        print("")
        print("Training complete!")
