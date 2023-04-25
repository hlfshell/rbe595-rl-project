import torch
from torch import Tensor
from torch.distributions import Normal, MultivariateNormal
from torch.nn import Sequential, Module, Linear
from torch.nn import Tanh, LeakyReLU, Softmax
import numpy as np
from torch import nn
import torch.nn.functional as F

from typing import Union, List


class PPOPoseActor(Module):
    def __init__(
        self,
        objects_spawned: int,
        control_type: str,
    ):
        self.objects_spawned = objects_spawned

        if control_type not in ["ee", "joint", "pendulum"]:
            raise Exception(
                'Control type must be specified as either "ee" or "joint" control'
            )
        self.control_type = control_type

        super(PPOPoseActor, self).__init__()

        self.__initialize_model__()

    def __initialize_model__(self):
        # Since we're working with a continuous action space we're going to
        # use a normal distribution to sample from. This will be our
        # output layer

        # Determine our input size. It is defined by the number of spanwed objects
        # at the start. Ultimately, it is # of (objects * 18) + 13
        if self.control_type == "pendulum":
            self.input_size = 3
        else:
            self.input_size = (self.objects_spawned * 18) + 13

        # Determine our output size
        if self.control_type == "ee":
            # We have 4 outputs - xyz and gripper torque
            self.output_size = 4
        elif self.control_type == "pendulum":
            self.output_size = 1
        else:
            # We have 8 outputs - torques for each of 7 joints and the gripper
            self.output_size = 8

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
        # self.layer1 = nn.Linear(self.input_size, 64)
        # self.layer2 = nn.Linear(64, 64)
        # self.layer3 = nn.Linear(64, self.output_size)
        self.cov_var = torch.full(size=(self.output_size,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def forward(self, input: Union[np.ndarray, Tensor, List]) -> MultivariateNormal:
        if isinstance(input, np.ndarray):
            input_tensor: Tensor = torch.from_numpy(input.astype("float32"))
        elif type(input) is list:
            input_tensor: Tensor = torch.from_numpy(np.array(input).astype("float32"))
        else:
            input_tensor = input

        # activation1 = F.relu(self.layer1(input_tensor))
        # activation2 = F.relu(self.layer2(activation1))
        # output = self.layer3(activation2)

		# return output
        output = self.model(input_tensor)

        distribution: MultivariateNormal = MultivariateNormal(output, self.cov_mat)

        return distribution

    def save(self, filepath: str):
        torch.save({
            "model": self.model.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        data = torch.load(filepath)
        self.model.load_state_dict(data["model"])


class PPOPoseCritic(Module):
    def __init__(
        self,
        objects_spawned: int,
        control_type: str,
    ):
        self.objects_spawned = objects_spawned

        if control_type not in ["ee", "joint", "pendulum"]:
            raise Exception(
                'Control type must be specified as either "ee" or "joint" control'
            )
        self.control_type = control_type

        super(PPOPoseCritic, self).__init__()

        self.__initialize_model__()

    def __initialize_model__(self):
        # Our score can be unbounded as a value from
        # some -XXX, +XXX, so we don't scale it w/ an activation
        # function
        if self.control_type == "pendulum":
            input_size = 3
        else:
            input_size = (self.objects_spawned * 18) + 13

        self.model = Sequential(
            Linear(input_size, 128),
            LeakyReLU(),
            Linear(128, 64),
            LeakyReLU(),
            Linear(64, 32),
            LeakyReLU(),
            Linear(32, 1),
        )
        # self.layer1 = nn.Linear(input_size, 64)
        # self.layer2 = nn.Linear(64, 64)
        # self.layer3 = nn.Linear(64, 1)

    def forward(self, input: np.ndarray) -> Tensor:
        if isinstance(input, np.ndarray):
            input_tensor: Tensor = torch.from_numpy(input.astype("float32"))
        elif type(input) is list:
            input_tensor: Tensor = torch.from_numpy(np.array(input).astype("float32"))
        else:
            input_tensor = input

        # activation1 = F.relu(self.layer1(input_tensor))
        # activation2 = F.relu(self.layer2(activation1))
        # output = self.layer3(activation2)

        # return output

        return self.model(input_tensor)

    def save(self, filepath: str):
        torch.save({
            "model": self.model.state_dict(),
        }, filepath)

    def load(self, filepath: str):
        data = torch.load(filepath)
        self.model.load_state_dict(data["model"])