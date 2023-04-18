from project.ppo.pose_model import PPOPoseActor, PPOPoseCritic
from panda_gym.envs.core import RobotTaskEnv
from torch.distributions import Normal
from torch import Tensor
import numpy as np

import torch
from pathlib import Path
from typing import Tuple, List
from collections import deque


class TrainingState:
    def __init__(
        self,
        timesteps: int,
        max_observation_memory: int,
        timesteps_per_batch: int,
    ):
        self.timesteps_total = timesteps
        self.timestep: int = 0
        self.timesteps_per_batch = timesteps_per_batch

        self.image_size = image_size
        self.images_per_input = images_per_input

        self.observations: List[np.array] = deque(maxlen=max_observation_memory)

        self.batch_actions: List[np.array] = deque(maxlen=max_observation_memory)

        # batch_obs = []             # batch observations
        # batch_acts = []            # batch actions
        # batch_log_probs = []       # log probs of each action
        # batch_rews = []            # batch rewards
        # batch_rtgs = []            # batch rewards-to-go
        # batch_lens = []            # episodic lengths in batch


class PPOPoseTrainer:
    def __init__(
        self,
        env: RobotTaskEnv,
        actor: PPOPoseActor,
        critic: PPOPoseCritic,
        total_timesteps: int,
        timesteps_per_batch: int = 1,
        max_observation_memory: int = 1_000_000,
        save_folder: str = "inprogress/ppo",
    ):
        self.env = env
        self.actor = actor
        self.critic = critic

        self.save_folder = save_folder

        # Hyperparameters
        self.λ = 0.95
        self.γ = 0.99
        self.ε = 0.2
        self.learning_rate = 0.005
        self.total_timesteps = total_timesteps
        self.timesteps_per_batch = timesteps_per_batch
        self.max_observation_memory = max_observation_memory

        # Create our optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.learning_rate
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.learning_rate
        )

    def should_reload(self) -> bool:
        """
        should_reload checks to see if there exists checkpoints to load from;
        if so, it will return True
        """
        return Path(self.save_folder).exists()

    def load_trainer(self):
        """
        load_trainer_state will reload all necessary checkpoints and memory training
        from the last checkpoint possible. This is not just the model weights,
        but also state memories and training loss progress, etc.
        """
        pass

    def calculate_advantage(
        self, rewards: List[int], qs: List[float], terminated: List[bool]
    ) -> Tuple[List[float], List[float]]:
        """
        calculate_advantage calculates the advantage for each timestep in the batch
        """
        returns: List[float] = []
        gae: float = 0

        # Calculate the advantage for each timestep
        # Note that bool * int = 0 or int as bool is converted to 0 or 1
        for index in reversed(range(len(rewards))):
            delta = (
                rewards[index] + self.λ * qs[index + 1] * terminated[index] - qs[index]
            )
            # γ is our smoothing factor, λ is our discount factor
            gae = delta + self.γ * self.λ * terminated[index] * gae
            # Append the advantage to the front of the list
            returns.insert(0, gae + qs[index])

        # Calculate the mean and standard deviation of the advantage
        advantage = np.array(returns) - qs[:-1]
        mean = np.mean(advantage)
        std = np.std(advantage)

        # The 1e-10 is to prevent division by zero
        normalized_advantage = (advantage - mean) / (std + 1e-10)

        return returns, normalized_advantage

    def calculate_rewards_to_go(
        self, rewards: List[int], episode_lengths: List[bool]
    ) -> np.array:
        """
        calculate_rewards_to_go calculates the rewards to go for each timestep
        in the batch
        """
        rtgs: List[float] = []

        for episode_length in reversed(episode_lengths):
            # episode_length is the number of episodes we need to isolate
            # from the end pf the rewards list
            episode_rewards = rewards[-episode_length:]

            # Now calculate rewards-to-go backwards from the terminal state
            # to the original state
            discounted_reward: float = 0
            for reward in reversed(episode_rewards):
                discounted_reward = reward + (self.λ * discounted_reward)
                # We insert here to append to the front, since we're working
                # backingwards on our episodes
                rtgs.insert(0, discounted_reward)

        return rtgs

    def train(self):
        self.env.reset()
        # observation = self.env.get_obs()
        observations: List[np.array] = []
        action_distributions: List[Normal] = []
        actions: List[np.array] = []
        episode_lengths: List[int] = []
        rewards: List[int] = []
        qs: List[float] = []
        terminateds: List[bool] = []
        # Batch data
        # batch_obs = []             # batch observations
        # batch_acts = []            # batch actions
        # batch_log_probs = []       # log probs of each action
        # batch_rews = []            # batch rewards
        # batch_rtgs = []            # batch rewards-to-go
        # batch_lens = []            # episodic lengths in batch

        # Number of timesteps run so far this batch
        # t = 0 while t < self.timesteps_per_batch:  # Rewards this episode
        # ep_rews = []  obs = self.env.reset()
        # done = False  for ep_t in range(self.max_timesteps_per_episode):
        #     # Increment timesteps ran this batch so far
        #     t += 1    # Collect observation
        #     batch_obs.append(obs)    action, log_prob = self.get_action(obs)
        #     obs, rew, done, _ = self.env.step(action)

        #     # Collect reward, action, and log prob
        #     ep_rews.append(rew)
        #     batch_acts.append(action)
        #     batch_log_probs.append(log_prob)    if done:
        #     break  # Collect episodic length and rewards
        # batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
        # batch_rews.append(ep_rews)

        tpb = 0  # timesteps per this batch
        steps_per_episode = 0
        while True:
            tpb += 1
            steps_per_episode += 1

            observation = self.env.get_obs()
            action_distribution = self.actor(observation)
            action = action_distribution.sample().numpy()
            observation, reward, terminated, _, _ = self.env.step(action)
            observation = observation["observation"]

            observations.append(observation)
            action_distributions.append(action_distribution)
            actions.append(action)
            rewards.append(reward)
            q: Tensor = self.critic(observation)
            qs.append(q.item())
            terminateds.append(terminated)

            if terminated:
                self.env.reset()

                episode_lengths.append(steps_per_episode)
                steps_per_episode = 0

                # If we have finished the episode and have the number
                # of timesteps we need for a batch, we stop our
                # collection and move on to training
                if tpb >= self.timesteps_per_batch:
                    break

        # Calculate our discounted rewards and advantage and normalized advantage
        # discounted_rewards, normalized_advantage = self.calculate_advantage(
        #     rewards, qs, terminateds
        # )

        # discounted_rewards = self.calculate_rewards_to_go(rewards, episode_lengths)
        # advantage = self.critic_model(observations).detach()

        # We only want timesteps_per_batch timesteps, but since we needed
        # to wait until we had a terminated episode at the end, we likely
        # overshot this. Drop the extra episodes from all recorded data
        observations = observations[0 : self.timesteps_per_batch]
        action_distributions = action_distributions[0 : self.timesteps_per_batch]
        actions = actions[0 : self.timesteps_per_batch]
        rewards = rewards[0 : self.timesteps_per_batch]
        terminateds = terminateds[0 : self.timesteps_per_batch]

        V = self.critic(observations)
        discounted_rewards = self.calculate_rewards_to_go(rewards, episode_lengths)
        A_k = Tensor(discounted_rewards) - V.detach()
        normalized_advantage = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        # Calculate the advantage for these sessions
        # returns, advantages = self.calculate_advantage(rewards, qs, terminated)

        # Get the log probabilities of our action_distrbutions - the
        # log probability is easier to work with mathematically for
        # gradient descent and fixes some issues during training
        print("here", len(action_distributions))
        log_probs: List[Tensor] = []
        for index in range(len(action_distributions)):
            distribution = action_distributions[index]
            action = Tensor(actions[index])
            log_probs.append(distribution.log_prob(action))
        # log_probs: List[Tensor] = [
        #     distribution.log_prob() for distribution in action_distributions
        # ]

        # We are now doing N epochs of training
        N = 5  # TODO make this a hyper parameter
        for _ in range(N):
            # We need our log_probabilities for our current epoch and our current
            # batch
            # TODO - make this a hyperparameter
            batch_size = 32
            current_log_probs: Tensor = Tensor()
            for batch in range(0, len(observations), batch_size):
                distributions = self.actor(observations[batch : batch + batch_size])
                actions = Tensor(actions[batch : batch + batch_size])
                log_probs = distributions.log_prob(actions)
                current_log_probs = torch.cat((current_log_probs, log_probs), dim=0)

            # We are calculating the ratio as defined by:
            #
            #   π_θ(a_t | s_t)
            #   --------------
            #   π_θ_k(a_t | s_t)
            # Where our originaly utilized log probabilities are
            # π_θ_k and our current model is creating π_θ. We
            # use the log probabilities and subtract, then raise
            # e to the power of the results to simplify the math
            # for back propagation/gradient descent
            ratios = torch.exp(current_log_probs - log_probs)

            # Now we need to calculate the actor's loss.
            # The losses  are either a clipped value or the ratio
            # of our advantage, whichever is smaller. This result
            # is clipped to prevent the loss from causing significant
            # swings in behaviour in the model resulting in a high
            # degree of instability. Finally, since it's a loss,
            # we take the negative of this value since we're trying
            # to minimize the loss with SGD while maximizing our
            # rewards. We the ntake the mean of this value to find
            # the loss of this entire batch.
            actor_loss = -torch.min(
                ratios * normalized_advantage,
                torch.clamp(ratios, 1 - self.ε, 1 + self.ε) * normalized_advantage,
            ).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            print("holy shit we made it this far")
            print(actor_loss)
            raise "holy hell"
