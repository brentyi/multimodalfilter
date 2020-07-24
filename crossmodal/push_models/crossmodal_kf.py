import torch
import torch.nn as nn

import diffbayes
import diffbayes.types as types
from fannypack.nn import resblocks

from ..base_models import (
    CrossmodalKalmanFilter,
    CrossmodalKalmanFilterMeasurementModel,
    CrossmodalKalmanFilterWeightModel,
)
from . import layers
from .dynamics import PushDynamicsModel
from .kf import PushKalmanFilterMeasurementModel
from .kf import PushKalmanFilter


class PushCrossmodalKalmanFilter(CrossmodalKalmanFilter):
    def __init__(self):
        """Initializes a particle filter for our Push task.
        """

        super().__init__(
            filter_models= [
                PushKalmanFilter(
                    dynamics_model=PushDynamicsModel,
                    measurement_model=PushKalmanFilterMeasurementModel(
                        modalities={"image"}
                    ),

                ),
                PushKalmanFilter(
                    dynamics_model=PushDynamicsModel,
                    measurement_model=PushKalmanFilterMeasurementModel(
                        modalities={"pos", "sensors"}
                    ),
                )
            ],
            crossmodal_weight_model=PushCrossmodalKalmanFilterWeightModel(state_dim=2),
            state_dim=2,
        )

class PushMeasurementCrossmodalKalmanFilter(PushKalmanFilter):
    def __init__(self):
        """Initializes a particle filter for our Push task.
        """

        super().__init__(
            dynamics_model=PushDynamicsModel,
            measurement_model=CrossmodalKalmanFilterMeasurementModel(
                measurement_models=[
                    PushKalmanFilterMeasurementModel(modalities={"image"}),
                    PushKalmanFilterMeasurementModel(modalities={"pos", "sensors"})
                ],
                crossmodal_weight_model=PushCrossmodalKalmanFilterWeightModel(state_dim=2),
                state_dim=2,
            ),
        )


class PushCrossmodalKalmanFilterWeightModel(CrossmodalKalmanFilterWeightModel):
    def __init__(self, units: int = 64, state_dim: int = 2):
        modality_count = 2
        super().__init__(modality_count=modality_count, state_dim=state_dim)

        self.observation_image_layers = layers.observation_image_layers(units)
        self.observation_pos_layers = layers.observation_pos_layers(units)
        self.observation_sensors_layers = layers.observation_sensors_layers(units)

        # Initial fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(units * 3, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
            nn.Linear(units, modality_count*self.state_dim),
            nn.Sigmoid(),
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
        assert output.shape == (N, self.modality_count * self.state_dim)

        state_weights = output.reshape(self.modality_count, N, self.state_dim)

        state_covariances = []
        # todo: do non-diagnonal terms
        for i in range(self.modality_count):
            state_covariances.append(
                torch.diag_embed(1. / (state_weights[i, :, :] + 1e-9), offset=0, dim1=-2, dim2=-1)
            )
        state_covariances = torch.stack(state_covariances)

        return state_weights, state_covariances