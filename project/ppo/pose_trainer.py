from __future__ import annotations
import pickle
from project.ppo.pose_model import PPOPoseActor, PPOPoseCritic
from panda_gym.envs.core import RobotTaskEnv
from torch import Tensor
from torch.nn import MSELoss
import numpy as np

import torch
from typing import Tuple, List
import sys

import matplotlib.pyplot as plt

EpisodeMemory: Tuple[List[np.array], List[np.array], List[np.array], List[float]]


class Trainer:
    def __init__(
        self,
        env: RobotTaskEnv,
        actor: PPOPoseActor,
        critic: PPOPoseCritic,
        timesteps: int,
        timesteps_per_batch: int,
        max_timesteps_per_episode: int,
        γ: float = 0.99,
        ε: float = 0.2,
        # α: float = 0.005,
        # α: float = 0.01,
        α: float = 3e-4,
        training_cycles_per_batch: int = 5,
        save_every_x_timesteps: int = 50_000,
    ):
        self.env = env
        self.actor = actor
        self.critic = critic

        self.timesteps = timesteps
        self.current_timestep = 0
        self.max_timesteps_per_episode = max_timesteps_per_episode
        self.timesteps_per_batch = timesteps_per_batch
        self.save_every_x_timesteps = save_every_x_timesteps
        self.last_save = 0

        # Hyperparameters
        self.γ = γ
        self.ε = ε
        self.α = α
        self.training_cycles_per_batch = training_cycles_per_batch

        # Memory
        self.total_rewards: List[float] = []
        self.terminal_timesteps: List[int] = []
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.α)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.α)

        # Initialize some status tracking memory
        self.previous_print_length: int = 0
        self.current_action = "Initializing"
        self.last_save: int = 0

    def print_status(self):
        latest_reward = 0.0
        average_reward = 0.0
        best_reward = 0.0
        latest_actor_loss = 0.0
        avg_actor_loss = 0.0
        latest_critic_loss = 0.0
        avg_critic_loss = 0.0
        recent_change = 0.0

        if len(self.total_rewards) > 0:
            latest_reward = self.total_rewards[-1]

            last_n_episodes = 100
            average_reward = np.mean(self.total_rewards[-last_n_episodes:])

            episodes = [
                i
                for i in range(
                    len(self.total_rewards[-last_n_episodes:]),
                    min(last_n_episodes, 0),
                    -1,
                )
            ]
            coefficients = np.polyfit(
                episodes,
                self.total_rewards[-last_n_episodes:],
                1,
            )
            recent_change = coefficients[0]

            best_reward = max(self.total_rewards)

        if len(self.actor_losses) > 0:
            avg_count = 3 * self.timesteps_per_batch
            latest_actor_loss = self.actor_losses[-1]
            avg_actor_loss = np.mean(self.actor_losses[-avg_count:])
            latest_critic_loss = self.critic_losses[-1]
            avg_critic_loss = np.mean(self.critic_losses[-avg_count:])

        msg = f"""
            =========================================
            Timesteps: {self.current_timestep:,} / {self.timesteps:,} ({round((self.current_timestep/self.timesteps)*100, 4)}%)
            Episodes: {len(self.total_rewards):,}
            Currently: {self.current_action}
            Latest Reward: {round(latest_reward)}
            Latest Avg Rewards: {round(average_reward)}
            Recent Change: {round(recent_change, 2)}
            Best Reward: {round(best_reward, 2)}
            Latest Actor Loss: {round(latest_actor_loss, 4)}
            Avg Actor Loss: {round(avg_actor_loss, 4)}
            Latest Critic Loss: {round(latest_critic_loss, 4)}
            Avg Critic Loss: {round(avg_critic_loss, 4)}
            =========================================
        """

        # We print to STDERR as a hack to get around the noisy pybullet
        # environment. Hacky, but effective if paired w/ 1> /dev/null
        print(msg, file=sys.stderr)

    def create_plot(self, filepath: str):
        last_n_episodes = 10

        episodes = [i + 1 for i in range(len(self.total_rewards))]
        averages = [
            np.mean(self.total_rewards[i - last_n_episodes : i])
            for i in range(len(self.total_rewards))
        ]
        trend_data = np.polyfit(episodes, self.total_rewards, 1)
        trendline = np.poly1d(trend_data)

        plt.scatter(
            episodes, self.total_rewards, color="green"
        )  # , linestyle='None', marker='o', color='green')
        plt.plot(episodes, averages, linestyle="solid", color="red")
        plt.plot(episodes, trendline(episodes), linestyle="--", color="blue")

        plt.title("Rewards per episode")
        plt.ylabel("Reward")
        plt.xlabel("Episode")
        plt.savefig(filepath)

    def save(self, directory: str):
        """
        save will save the models, state, and any additional
        data to the given directory
        """
        self.last_save = self.current_timestep

        self.actor.save(f"{directory}/actor.pth")
        self.critic.save(f"{directory}/critic.pth")
        self.create_plot(f"{directory}/rewards.png")

        # Now save the trainer's state data
        data = {
            "timesteps": self.timesteps,
            "current_timestep": self.current_timestep,
            "max_timesteps_per_episode": self.max_timesteps_per_episode,
            "timesteps_per_batch": self.timesteps_per_batch,
            "save_every_x_timesteps": self.save_every_x_timesteps,
            "γ": self.γ,
            "ε": self.ε,
            "α": self.α,
            "training_cycles_per_batch": self.training_cycles_per_batch,
            "total_rewards": self.total_rewards,
            "terminal_timesteps": self.terminal_timesteps,
            "actor_losses": self.actor_losses,
            "critic_losses": self.critic_losses,
        }
        pickle.dump(data, open(f"{directory}/state.data", "wb"))

    def load(self, directory: str):
        """
        Load will load the models, state, and any additional
        data from the given directory
        """
        # Load our models first; they're the simplest
        self.actor.load(f"{directory}/actor.pth")
        self.critic.load(f"{directory}/critic.pth")

        data = pickle.load(open(f"{directory}/state.data", "rb"))

        self.timesteps = data["timesteps"]
        self.current_timestep = data["current_timestep"]
        self.last_save = self.current_timestep
        self.max_timesteps_per_episode = data["max_timesteps_per_episode"]
        self.timesteps_per_batch = data["timesteps_per_batch"]
        self.save_every_x_timesteps = data["save_every_x_timesteps"]

        # Hyperparameters
        self.γ = data["γ"]
        self.ε = data["ε"]
        self.α = data["α"]
        self.training_cycles_per_batch = data["training_cycles_per_batch"]

        # Memory
        self.total_rewards = data["total_rewards"]
        self.terminal_timesteps = data["terminal_timesteps"]
        self.actor_losses = data["actor_losses"]
        self.critic_losses = data["critic_losses"]

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.α)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.α)

    def run_episode(self) -> EpisodeMemory:
        """
        run_episode runs a singular episode and returns the results
        """
        observation, _ = self.env.reset()
        if isinstance(observation, dict):
            observation = observation["observation"]

        timesteps = 0
        observations: List[np.ndarray] = []
        actions: List[np.ndarray] = []
        log_probabilities: List[float] = []
        rewards: List[float] = []

        while True:
            timesteps += 1

            observations.append(observation)
            action_distribution = self.actor(observation)
            action = action_distribution.sample()
            log_probability = action_distribution.log_prob(action).detach().numpy()
            action = action.detach().numpy()
            observation, reward, terminated, _, _ = self.env.step(action)
            if isinstance(observation, dict):
                observation = observation["observation"]

            actions.append(action)
            log_probabilities.append(log_probability)
            rewards.append(reward)

            if timesteps >= self.max_timesteps_per_episode:
                terminated = True

            if terminated:
                break

        # Calculate the discounted rewards for this episode
        discounted_rewards: List[float] = self.calculate_discounted_rewards(rewards)

        # Get the terminal reward and record it for status tracking
        self.total_rewards.append(sum(rewards))

        return observations, actions, log_probabilities, discounted_rewards

    def rollout(self) -> EpisodeMemory:
        """
        rollout will perform a rollout of the environment and
        return the memory of the episode with the current
        actor model
        """
        observations: List[np.ndarray] = []
        log_probabilities: List[float] = []
        actions: List[float] = []
        rewards: List[float] = []

        while len(observations) < self.timesteps_per_batch:
            self.current_action = "Rollout"
            obs, chosen_actions, log_probs, rwds = self.run_episode()
            # Combine these arrays into our overall batch
            observations += obs
            actions += chosen_actions
            log_probabilities += log_probs
            rewards += rwds

            # Increment our count of timesteps
            self.current_timestep += len(obs)

            self.print_status()

        # We need to trim the batch memory to the batch size
        observations = observations[: self.timesteps_per_batch]
        actions = actions[: self.timesteps_per_batch]
        log_probabilities = log_probabilities[: self.timesteps_per_batch]
        rewards = rewards[: self.timesteps_per_batch]

        return observations, actions, log_probabilities, rewards

    def calculate_discounted_rewards(self, rewards: List[float]) -> List[float]:
        """
        calculated_discounted_rewards will calculate the discounted rewards
        of each timestep of an episode given its initial rewards and episode
        length
        """
        discounted_rewards: List[float] = []
        discounted_reward: float = 0
        for reward in reversed(rewards):
            discounted_reward = reward + (self.γ * discounted_reward)
            # We insert here to append to the front as we're calculating
            # backwards from the end of our episodes
            discounted_rewards.insert(0, discounted_reward)

        return discounted_rewards

    def calculate_normalized_advantage(
        self, observations: Tensor, rewards: Tensor
    ) -> Tensor:
        """
        calculate_normalized_advantage will calculate the normalized
        advantage of a given batch of episodes
        """
        V = self.critic(observations).detach().squeeze()

        # Now we need to calculate our advantage and normalize it
        advantage = Tensor(np.array(rewards, dtype="float32")) - V
        normalized_advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        return normalized_advantage

    def training_step(
        self,
        observations: Tensor,
        actions: Tensor,
        log_probabilities: Tensor,
        rewards: Tensor,
        normalized_advantage: Tensor,
    ) -> Tuple[float, float]:
        """
        training_step will perform a single epoch of training for the
        actor and critic model

        Returns the loss for each model at the end of the step
        """
        # Get our output for the current actor given our log
        # probabilities
        current_action_distributions = self.actor(observations)
        current_log_probabilities = current_action_distributions.log_prob(actions)

        # We are calculating the ratio as defined by:
        #
        #   π_θ(a_t | s_t)
        #   --------------
        #   π_θ_k(a_t | s_t)
        #
        # ...where our originaly utilized log probabilities
        # are π_θ_k and our current model is creating π_θ. We
        # use the log probabilities and subtract, then raise
        # e to the power of the results to simplify the math
        # for back propagation/gradient descent.
        # Note that we have a log probability matrix of shape
        # (batch size, number of actions), where we're expecting
        # (batch size, 1). We sum our logs as the log(A + B) =
        # log(A) + log(B).
        # log_probabilities = Tensor(np.array(log_probabilities, dtype=np.float32))
        # log_probabilities = torch.sum(log_probabilities, dim=-1)
        # current_log_probabilities = torch.sum(current_log_probabilities, dim=-1)
        ratio = torch.exp(current_log_probabilities - log_probabilities)

        # Now we calculate the actor loss for this step
        actor_loss = -torch.min(
            ratio * normalized_advantage,
            torch.clamp(ratio, 1 - self.ε, 1 + self.ε) * normalized_advantage,
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

        return actor_loss.item(), critic_loss.item()

    def train(self):
        """
        train will train the actor and critic models with the
        given training state
        """
        while self.current_timestep <= self.timesteps:
            # Perform a rollout to get our next training
            # batch
            observations, actions, log_probabilities, rewards = self.rollout()

            # Finally, convert these to numpy arrays and then to tensors
            observations = Tensor(np.array(observations, dtype=np.float32))
            actions = Tensor(np.array(actions, dtype=np.float32))
            log_probabilities = Tensor(np.array(log_probabilities, dtype=np.float32))
            rewards = Tensor(np.array(rewards, dtype=np.float32))

            # Perform our training steps
            for c in range(self.training_cycles_per_batch):
                self.current_action = (
                    f"Training Cycle {c+1}/{self.training_cycles_per_batch}"
                )
                self.print_status()

                # Calculate our losses
                normalized_advantage = self.calculate_normalized_advantage(
                    observations, rewards
                )
                actor_loss, critic_loss = self.training_step(
                    observations,
                    actions,
                    log_probabilities,
                    rewards,
                    normalized_advantage,
                )
                self.actor_losses.append(actor_loss)
                self.critic_losses.append(critic_loss)

            # Every X timesteps, save our current status
            if self.current_timestep - self.last_save >= self.save_every_x_timesteps:
                self.current_action = "Saving"
                self.print_status()
                self.save("training")

        print("")
        print("Training complete!")

        # Save our results
        self.save("training")
