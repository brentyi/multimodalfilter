import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import diffbayes
import diffbayes.types as types
from fannypack.nn import resblocks

from ..base_models import (
    CrossmodalKalmanFilter,
    CrossmodalKalmanFilterMeasurementModel,
    CrossmodalKalmanFilterWeightModel,
)
from ..tasks import DoorTask
from . import layers
from .dynamics import DoorDynamicsModel
from .kf import DoorKalmanFilter, DoorKalmanFilterMeasurementModel


class DoorCrossmodalKalmanFilter(CrossmodalKalmanFilter, DoorTask.Filter):
    def __init__(self, know_image_blackout=False):
        """Initializes a particle filter for our door task.
        """

        super().__init__(
            filter_models=[
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
                ),
            ],
            crossmodal_weight_model=DoorCrossmodalKalmanFilterWeightModel(state_dim=3),
            state_dim=3,
        )

        self.know_image_blackout = know_image_blackout

    def forward(
            self, *, observations: types.ObservationsTorch,
            controls: types.ControlsTorch,
    ) -> types.StatesTorch:
        N, _ = controls.shape
        device = controls.device

        if self.know_image_blackout:

            blackout_indices = torch.sum(torch.abs(
                observations['image'].reshape((N, -1))), dim=1) < 1e-8

            if torch.sum(blackout_indices) == 0 or \
                    np.sum(self._enabled_models) < len(self._enabled_models):
                return super().forward(observations=observations, controls=controls)

            unimodal_states, unimodal_covariances = self.calculate_unimodal_states(observations,
                                                                                   controls)
            raw_state_weights = self.crossmodal_weight_model(observations=observations)
            image_weight = raw_state_weights[0]
            force_weight = raw_state_weights[1]

            mask_shape = (N, 1)
            mask = torch.ones(mask_shape, device=device)
            mask[blackout_indices] = 0

            image_beta_new = torch.zeros(mask_shape, device=device)
            image_beta_new[blackout_indices] = 1e-9
            image_weight = image_beta_new + mask * image_weight

            force_beta_new = torch.zeros(mask_shape, device=device)
            force_beta_new[blackout_indices] = 1. - 1e-9
            force_weight = force_beta_new + mask * force_weight

            state_weights = torch.stack([image_weight, force_weight])
            assert state_weights.shape == (np.sum(self._enabled_models), N, self.state_dim)

            weighted_states, weighted_covariances = self.calculate_weighted_states(state_weights,
                                                                                   unimodal_states,
                                                                                   unimodal_covariances)

            self.weighted_covariances = weighted_covariances

            return weighted_states

        return super().forward(observations=observations, controls=controls)


class DoorCrossmodalKalmanFilterWeightModel(CrossmodalKalmanFilterWeightModel):
    def __init__(self, units: int = 64, state_dim: int = 2):
        modality_count = 2
        super().__init__(modality_count=modality_count, state_dim=state_dim)

        self.weighting_types = ["softmax", "absolute"]

        self.observation_image_layers = layers.observation_image_layers(units)
        self.observation_pos_layers = layers.observation_pos_layers(units)
        self.observation_sensors_layers = layers.observation_sensors_layers(units)
        self.weighting_type = "softmax"

        assert self.weighting_type in self.weighting_types

        self.fusion_layers = nn.Sequential(
            nn.Linear(units * 3, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
            nn.Linear(units, modality_count * self.state_dim),
        )


    def forward(self, *, observations: types.ObservationsTorch) -> torch.Tensor:
        """Compute modality weights.

        Args:
            observations (types.ObservationsTorch): Model observations.

        Returns:
            torch.Tensor: Computed weights. Shape should be `(N, modality_count)`.
        """
        assert self.weighting_type in self.weighting_types

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
            state_weights = F.softmax(state_weights, dim=0)

        state_weights = state_weights / (torch.sum(state_weights, dim=0) + 1e-9)

        return state_weights


class DoorMeasurementCrossmodalKalmanFilter(DoorKalmanFilter):
    def __init__(self):
        """Initializes a crossmodal kalman filter for our door task.
        """

        super().__init__(
            dynamics_model=DoorDynamicsModel(),
            measurement_model=CrossmodalKalmanFilterMeasurementModel(
                measurement_models=[
                    DoorKalmanFilterMeasurementModel(modalities={"image"}),
                    DoorKalmanFilterMeasurementModel(modalities={"pos", "sensors"}),
                ],
                crossmodal_weight_model=DoorCrossmodalKalmanFilterWeightModel(
                    state_dim=3
                ),
                state_dim=3,
            ),
        )
