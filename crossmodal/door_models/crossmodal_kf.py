import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


import diffbayes
import diffbayes.types as types
from fannypack.nn import resblocks

from ..base_models import (
    CrossmodalKalmanFilter,
    CrossmodalKalmanFilterMeasurementModel,
    CrossmodalKalmanFilterWeightModel,
)
from . import layers
from .dynamics import DoorDynamicsModel
from .kf import DoorKalmanFilterMeasurementModel
from .kf import DoorKalmanFilter


class DoorCrossmodalKalmanFilter(CrossmodalKalmanFilter):
    def __init__(self, know_image_blackout=False):
        """Initializes a particle filter for our door task.
        """

        super().__init__(
            filter_models= [
                DoorKalmanFilter(
                    dynamics_model=DoorDynamicsModel(),
                    measurement_model=DoorKalmanFilterMeasurementModel(
                        modalities={"image"}
                    ),

                ),
                DoorKalmanFilter(
                    dynamics_model=DoorDynamicsModel(),
                    measurement_model=DoorKalmanFilterMeasurementModel(
                        modalities={"pos", "sensors"}
                    ),
                )
            ],
            crossmodal_weight_model=DoorCrossmodalKalmanFilterWeightModel(state_dim=3),
            state_dim=3,
        )

        self.know_image_blackout = know_image_blackout


class DoorCrossmodalKalmanFilterWeightModel(CrossmodalKalmanFilterWeightModel):
    def __init__(self, units: int = 64, state_dim: int = 2,
                 know_image_blackout=False):
        modality_count = 2
        super().__init__(modality_count=modality_count, state_dim=state_dim)

        weighting_types = ["sigmoid", "softmax", "absolute"]

        self.observation_image_layers = layers.observation_image_layers(units)
        self.observation_pos_layers = layers.observation_pos_layers(units)
        self.observation_sensors_layers = layers.observation_sensors_layers(units)
        self.weighting_type = "softmax"

        assert self.weighting_type in weighting_types

        if self.weighting_type == "sigmoid":
        # Initial fusion layers
            self.fusion_layers = nn.Sequential(
                nn.Linear(units * 3, units),
                nn.ReLU(inplace=True),
                resblocks.Linear(units),
                nn.Linear(units, modality_count*self.state_dim),
                nn.Sigmoid(),
            )
        else:
            self.fusion_layers = nn.Sequential(
                nn.Linear(units * 3, units),
                nn.ReLU(inplace=True),
                resblocks.Linear(units),
                nn.Linear(units, modality_count * self.state_dim),
            )

        self.know_image_blackout = know_image_blackout



    def forward(self, *, observations: types.ObservationsTorch) -> torch.Tensor:
        """Compute modality weights.

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
        assert output.shape == (N, self.modality_count * self.state_dim)

        state_weights = output.reshape(self.modality_count, N, self.state_dim)

        if self.weighting_type == "absolute":
            state_weights = torch.abs(state_weights)
        elif self.weighting_type == "softmax":
            state_weights = F.softmax(
                state_weights,
                dim=0
            )

        state_weights = state_weights / (torch.sum((state_weights), dim=0) + 1e-9)

        return state_weights


class DoorMeasurementCrossmodalKalmanFilter(DoorKalmanFilter):
    def __init__(self):
        """Initializes a particle filter for our door task.
        """

        super().__init__(
            dynamics_model=DoorDynamicsModel(),
            measurement_model=CrossmodalKalmanFilterMeasurementModel(
                measurement_models=[
                    DoorKalmanFilterMeasurementModel(modalities={"image"}),
                    DoorKalmanFilterMeasurementModel(modalities={"pos", "sensors"})
                ],
                crossmodal_weight_model=DoorCrossmodalKalmanFilterWeightModel(state_dim=3),
                state_dim=3,
            ),
        )