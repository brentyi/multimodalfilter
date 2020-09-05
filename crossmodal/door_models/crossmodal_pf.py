import diffbayes
import diffbayes.types as types
import numpy as np
import torch
import torch.nn as nn
from fannypack.nn import resblocks

from ..base_models import (
    CrossmodalParticleFilterMeasurementModel,
    CrossmodalWeightModel,
)
from ..tasks import DoorTask
from . import layers
from .dynamics import DoorDynamicsModelBrent
from .pf import DoorMeasurementModel


class DoorCrossmodalParticleFilter(diffbayes.base.ParticleFilter, DoorTask.Filter):
    def __init__(self, know_image_blackout: bool = False):
        """Initializes a particle filter for our door task.
        """

        super().__init__(
            dynamics_model=DoorDynamicsModelBrent(),
            measurement_model=CrossmodalParticleFilterMeasurementModel(
                measurement_models=[
                    DoorMeasurementModel(modalities={"image"}),
                    DoorMeasurementModel(modalities={"pos", "sensors"}),
                ],
                crossmodal_weight_model=DoorCrossmodalWeightModel(
                    know_image_blackout=know_image_blackout
                ),
                state_dim=3,
            ),
            num_particles=30,
        )

    def train(self, mode: bool = True):
        """Adjust particle count based on train vs eval mode.
        """
        self.num_particles = 30 if mode else 300
        super().train(mode)


class DoorCrossmodalParticleFilterSeq5(DoorCrossmodalParticleFilter, DoorTask.Filter):
    """Blackout-aware version of crossmodal particle filter: should only be used on
    seq5 dataset.
    """

    def __init__(self):
        super().__init__(know_image_blackout=True)


class DoorCrossmodalWeightModel(CrossmodalWeightModel):
    def __init__(self, know_image_blackout: bool, units: int = 64):
        modality_count = 2
        super().__init__(modality_count=modality_count)

        self.know_image_blackout = know_image_blackout

        self.observation_image_layers = layers.observation_image_layers(units)
        self.observation_pos_layers = layers.observation_pos_layers(units)
        self.observation_sensors_layers = layers.observation_sensors_layers(units)

        # Initial fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(units * 3, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
            resblocks.Linear(units),
            resblocks.Linear(units),
            nn.Linear(units, modality_count),
            # nn.LogSigmoid() # <- This drops performance slightly
        )

    def forward(self, *, observations: types.ObservationsTorch) -> torch.Tensor:
        """Compute log-modality weights.

        Args:
            observations (types.ObservationsTorch): Model observations.

        Returns:
            torch.Tensor: Computed weights. Shape should be `(N, modality_count)`.
        """

        N, _ = observations["gripper_pos"].shape

        # Construct observations feature vector
        # (N, obs_dim)
        obs = []
        obs.append(self.observation_image_layers(observations["image"][:, None, :, :]))
        obs.append(self.observation_pos_layers(observations["gripper_pos"]))
        obs.append(self.observation_sensors_layers(observations["gripper_sensors"]))
        observation_features = torch.cat(obs, dim=1)

        # Propagate through fusion layers
        output = self.fusion_layers(observation_features)
        assert output.shape == (N, self.modality_count)

        # Know image blackout
        if self.know_image_blackout:
            blackout_indices = (
                torch.sum(torch.abs(observations["image"].reshape((N, -1))), dim=1)
                < 1e-8
            )
            output[blackout_indices, 0] -= np.inf

        return output
