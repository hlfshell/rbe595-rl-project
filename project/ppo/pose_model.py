import torch
from torch import Tensor
from torch.distributions import Normal
from torch.nn import Sequential, Module, Linear
from torch.nn import Tanh, LeakyReLU, Softmax
import numpy as np

from typing import Union


class PPOPoseActor(Module):
    def __init__(
        self,
        objects_spawned: int,
        control_type: str,
    ):
        self.objects_spawned = objects_spawned

        if control_type not in ["ee", "joint"]:
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
        self.input_size = (self.objects_spawned * 18) + 13

        # Determine our output size
        if self.control_type == "ee":
            # We have 4 outputs - xyz and gripper torque
            self.output_size = 4
        else:
            # We have 8 outputs - torques for each of 7 joints and the gripper
            self.output_size = 8

        # Create our model head - input -> some output layer before splitting to
        # different outputs for mu and sigma
        self.model_head = Sequential(
            Linear(self.input_size, 512),
            LeakyReLU(),
            Linear(512, 256),
            LeakyReLU(),
            Linear(256, 128),
            LeakyReLU(),
        )

        # Mu gives us a value -1 to 1, so tanh is used
        self.mu_model = Sequential(
            Linear(128, 64),
            LeakyReLU(),
            Linear(64, 32),
            LeakyReLU(),
            Linear(32, self.output_size),
            Tanh(),
        )

        # Sigma needs to be values of 0 to 1, so we use softmax instead
        self.sigma_model = Sequential(
            Linear(128, 64),
            LeakyReLU(),
            Linear(64, 32),
            LeakyReLU(),
            Linear(32, self.output_size),
            Softmax(dim=0),
        )

    def forward(self, input: Union[np.ndarray, Tensor]) -> Normal:
        if isinstance(input, np.ndarray):
            input_tensor: Tensor = torch.from_numpy(input.astype("float32"))
        elif type(input) is list:
            input_tensor: Tensor = torch.from_numpy(np.array(input).astype("float32"))
        else:
            input_tensor = input

        # Put into our model head for the shared output
        model_head: Tensor = self.model_head(input_tensor)

        # Get our mu and sigma tensors
        mu_tensor: Tensor = self.mu_model(model_head)
        sigma_tensor: Tensor = self.sigma_model(model_head)

        # Create our normal distribution and sample for our output
        distribution: Normal = Normal(mu_tensor, sigma_tensor)

        return distribution

    def save(self, filepath: str):
        torch.save(
            {
                "model_head": self.model_head.state_dict(),
                "mu_model": self.mu_model.state_dict(),
                "sigma_model": self.sigma_model.state_dict(),
            },
            filepath,
        )

    def load(self, filepath: str):
        loadfile = torch.load(filepath)
        self.model_head.load_state_dict(loadfile["model_head"])
        self.mu_model.load_state_dict(loadfile["mu_model"])
        self.sigma_model.load_state_dict(loadfile["sigma_model"])


class PPOPoseCritic(Module):
    def __init__(
        self,
        objects_spawned: int,
        control_type: str,
    ):
        self.objects_spawned = objects_spawned

        if control_type not in ["ee", "joint"]:
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
        loadfile = torch.load(filepath)
        self.model.load_state_dict(loadfile["model"])
