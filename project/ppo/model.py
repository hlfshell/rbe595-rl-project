import torch
from torch import Tensor
from torch.distributions import MultivariateNormal
from torch.nn import Sequential, Module, Linear
from torch.nn import LeakyReLU
import numpy as np

from typing import Union, List, Any


class Actor(Module):
    def __init__(
        self,
        env: Any,
    ):
        self.input_size = env.observation_space["observation"].shape[0]
        self.output_size = env.action_space.shape[0]

        super(Actor, self).__init__()

        self.__initialize_model__()

    def __initialize_model__(self):
        # Create our model head - input -> some output layer before splitting to
        # different outputs for mu and sigma
        self.model = Sequential(
            Linear(self.input_size, 1024),
            LeakyReLU(),
            Linear(1024, 512),
            LeakyReLU(),
            Linear(512, 256),
            LeakyReLU(),
            Linear(256, 128),
            LeakyReLU(),
            Linear(128, 64),
            LeakyReLU(),
            Linear(64, 32),
            LeakyReLU(),
            Linear(32, self.output_size),
        )

        self.cov_var = torch.full(size=(self.output_size,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def forward(self, input: Union[np.ndarray, Tensor, List]) -> MultivariateNormal:
        if isinstance(input, np.ndarray):
            input_tensor: Tensor = torch.from_numpy(input.astype("float32"))
        elif type(input) is list:
            input_tensor: Tensor = torch.from_numpy(np.array(input).astype("float32"))
        else:
            input_tensor = input

        output = self.model(input_tensor)

        distribution: MultivariateNormal = MultivariateNormal(output, self.cov_mat)

        return distribution

    def save(self, filepath: str):
        torch.save(
            {
                "model": self.model.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str):
        data = torch.load(filepath)
        self.model.load_state_dict(data["model"])


class Critic(Module):
    def __init__(
        self,
        env: Any,
    ):
        self.input_size = env.observation_space["observation"].shape[0]

        super(Critic, self).__init__()

        self.__initialize_model__()

    def __initialize_model__(self):
        # Our score can be unbounded as a value from
        # some -XXX, +XXX, so we don't scale it w/ an activation
        # function

        self.model = Sequential(
            Linear(self.input_size, 128),
            LeakyReLU(),
            Linear(128, 64),
            LeakyReLU(),
            Linear(64, 32),
            LeakyReLU(),
            Linear(32, 1),
        )

    def forward(self, input: np.ndarray) -> Tensor:
        if isinstance(input, np.ndarray):
            input_tensor: Tensor = torch.from_numpy(input.astype("float32"))
        elif type(input) is list:
            input_tensor: Tensor = torch.from_numpy(np.array(input).astype("float32"))
        else:
            input_tensor = input

        return self.model(input_tensor)

    def save(self, filepath: str):
        torch.save(
            {
                "model": self.model.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str):
        data = torch.load(filepath)
        self.model.load_state_dict(data["model"])
