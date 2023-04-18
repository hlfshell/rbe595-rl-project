import torch
from torch import Tensor, stack
from torch.distributions import Normal
from torch.nn import Sequential, Conv2d, Module, Flatten, Linear
from torch.nn import Tanh, LeakyReLU, Softmax
import numpy as np

from collections import deque
from typing import Tuple, List
from math import floor


class PPOImageInput(deque):
    def __init__(self, image_size: Tuple[int, int], images: int = 4):
        super().__init__(self, maxlen=images)

        # Create a "blank" image for the initialization of this
        # deque
        for i in range(0, images):
            # self.deque.append(np.zeros(image_size + (3,)))
            self.append(np.zeros(image_size + (3,)))

    def get_tensor(self):
        tensors: List[Tensor] = []
        for image in self.deque:
            image: np.ndarray
            # We need to transpose the image so that channels are first
            tensor = torch.from_numpy(image.astype("float32").transpose(2, 0, 1))
            tensors.append(tensor)

        return stack(tensors)


class PPOImageActor(Module):
    def __init__(
        self,
        img_size: Tuple[int, int],
        control_type: str,
        images_per_input: int = 4,
        kernel_size: int = 5,
    ):
        self.img_size = img_size
        self.control_type = control_type
        if control_type not in ["joint", "ee"]:
            raise Exception(
                "Control type must be specified as either joint or ee control"
            )
        self.images_per_input = images_per_input
        self.kernel_size = 5
        # Our action space is determined on the control type. With ee it will
        # be the xyz and gripper torque changes. With joint control it will
        # instead be 8 outputs - torques for each of 7 joints and the gripper
        self.action_space_size = 4 if self.control_type == "ee" else 8

        super(PPOActor, self).__init__()

        self.__initialize_model__()

    def __initialize_model__(self):
        # Since we're working with a continuous action space we're going to
        # output a resulting mu and sigma to create the normal distribution
        # from which we'll sample our outputs.
        c1_size = calculate_conv2d_output_shape(
            self.img_size[1], self.img_size[0], (self.kernel_size, self.kernel_size)
        )
        c2_size = calculate_conv2d_output_shape(
            c1_size[1], c1_size[0], (self.kernel_size, self.kernel_size)
        )
        c3_size = calculate_conv2d_output_shape(
            c2_size[1], c2_size[0], (self.kernel_size, self.kernel_size)
        )
        flatten_size = 16 * c3_size[0] * c3_size[1]

        self.actor_head = Sequential(
            Conv2d(self.images_per_input * 3, 64, self.kernel_size),
            Conv2d(64, 32, self.kernel_size),
            Conv2d(32, 16, self.kernel_size),
            Flatten(),
            Linear(flatten_size, 128),
            LeakyReLU(),
        )
        # Output size of this model is expected to be

        # We need the mu to be calculated within our given range, which
        # is normalized to -1 to 1.
        self.mu_model = Sequential(
            Linear(128, 64),
            LeakyReLU(),
            Linear(64, self.action_space_size),
            Tanh(),
        )

        # Similarly, when we calculate the resulting sigma, we hope
        # we need values to fall within 0 to 1.
        self.sigma_model = Sequential(
            Linear(128, 64),
            LeakyReLU(),
            Linear(64, self.action_space_size),
            Softmax(),
        )

    def forward(self, input: PPOImageInput) -> Tensor:
        input_tensor = input.get_tensor()

        output = self.actor_head(input_tensor)
        mu = self.mu_model(output)
        sigma = self.sigma_model(output)
        distribution = Normal(mu, sigma)
        actions = distribution.sample()

        return actions

    def save(self, filepath: str):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str):
        self.load_state_dict(torch.load(filepath))


class PPOImageCritic(Module):
    def __init__(
        self,
        img_size: Tuple[int, int],
        control_type: str,
        images_per_input: int = 4,
        kernel_size: int = 5,
    ):
        self.img_size = img_size
        self.control_type = control_type
        if control_type not in ["joint", "ee"]:
            raise Exception(
                "Control type must be specified as either joint or ee control"
            )
        self.images_per_input = images_per_input
        self.kernel_size = kernel_size
        # Our action space is determined on the control type. With ee it will
        # be the xyz and gripper torque changes. With joint control it will
        # instead be 8 outputs - torques for each of 7 joints and the gripper
        self.action_space_size = 4 if self.control_type == "ee" else 8

        super(PPOImageCritic, self).__init__()

        self.__initialize_model__()

    def __initialize_model__(self):
        # Our score can be unbounded as a value from
        # some -XXX, +XXX, so we don't scale it w/ an activation
        # function
        c1_size = calculate_conv2d_output_shape(
            self.img_size[1], self.img_size[0], (self.kernel_size, self.kernel_size)
        )
        c2_size = calculate_conv2d_output_shape(
            c1_size[1], c1_size[0], (self.kernel_size, self.kernel_size)
        )
        flatten_size = 32 * c2_size[0] * c2_size[1]
        self.critic = Sequential(
            Conv2d(self.images_per_input * 3, 64, 5),
            Conv2d(64, 32, 5),
            Flatten(),
            Linear(flatten_size, 64),
            LeakyReLU(),
            Linear(64, 1),
        )

    def forward(self, input: PPOImageInput) -> Tensor:
        input_tensor = input.get_tensor()

        value = self.critic(input_tensor)

        return value

    def save(self, filepath: str):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str):
        self.load_state_dict(torch.load(filepath))


def calculate_conv2d_output_shape(
    height: int,
    width: int,
    kernel_size: Tuple[int, int] = (1, 1),
    stride: Tuple[int, int] = (1, 1),
    padding: Tuple[int, int] = (0, 0),
    dilation: Tuple[int, int] = (1, 1),
):
    h = floor(
        ((height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0])
        + 1
    )
    w = floor(
        ((width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1])
        + 1
    )

    return (h, w)
