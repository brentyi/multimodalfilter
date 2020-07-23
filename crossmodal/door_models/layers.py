import torch.nn as nn
import torch
from fannypack.nn import resblocks

state_dim = 3
control_dim = 7
obs_pos_dim = 3
obs_sensors_dim = 7


def state_layers(units: int) -> nn.Module:
    """Create a state encoder block.

    Args:
        units (int): # of hidden units in network layers.

    Returns:
        nn.Module: Encoder block.
    """
    return nn.Sequential(
        nn.Linear(state_dim, units), nn.ReLU(inplace=True), resblocks.Linear(units),
    )


def control_layers(units: int) -> nn.Module:
    """Create a control command encoder block.

    Args:
        units (int): # of hidden units in network layers.

    Returns:
        nn.Module: Encoder block.
    """
    return nn.Sequential(
        nn.Linear(control_dim, units), nn.ReLU(inplace=True), resblocks.Linear(units),
    )


def observation_image_layers(units: int) -> nn.Module:
    """Create an image encoder block.

    Args:
        units (int): # of hidden units in network layers.

    Returns:
        nn.Module: Encoder block.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
        nn.ReLU(inplace=True),
        resblocks.Conv2d(channels=32, kernel_size=3),
        nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
        nn.Flatten(),  # 32 * 32 * 8
        nn.Linear(8 * 32 * 32, units),
        nn.ReLU(inplace=True),
        resblocks.Linear(units),
    )


def observation_pos_layers(units: int) -> nn.Module:
    """Create an end effector position encoder block.

    Args:
        units (int): # of hidden units in network layers.

    Returns:
        nn.Module: Encoder block.
    """
    return nn.Sequential(
        nn.Linear(obs_pos_dim, units),
        nn.ReLU(inplace=True),
        resblocks.Linear(units),
    )


def observation_sensors_layers(units: int) -> nn.Module:
    """Create an F/T sensor encoder block.

    Args:
        units (int): # of hidden units in network layers.

    Returns:
        nn.Module: Encoder block.
    """
    return nn.Sequential(
        nn.Linear(obs_sensors_dim, units),
        nn.ReLU(inplace=True),
        resblocks.Linear(units),
    )
