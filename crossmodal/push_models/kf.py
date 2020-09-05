from typing import Set, cast

import diffbayes
import diffbayes.types as types
import torch
import torch.nn as nn
from fannypack.nn import resblocks

from ..tasks import PushTask
from . import layers
from .dynamics import PushDynamicsModel


class PushKalmanFilter(diffbayes.base.KalmanFilter, PushTask.Filter):
    def __init__(self, dynamics_model=None, measurement_model=None):
        """Initializes a particle filter for our door task.
        """
        if dynamics_model is None and measurement_model is None:
            super().__init__(
                dynamics_model=PushDynamicsModel(),
                measurement_model=PushKalmanFilterMeasurementModel(),
            )
        else:
            super().__init__(
                dynamics_model=dynamics_model, measurement_model=measurement_model,
            )


class PushKalmanFilterMeasurementModel(diffbayes.base.KalmanFilterMeasurementModel):
    def __init__(
        self,
        units: int = 64,
        modalities: Set[str] = {"image", "pos", "sensors"},
        add_R_noise: float = 1e-6,
        noise_R_tril: torch.Tensor = None,
    ):
        """Initializes a measurement model for our door task.
        """

        super().__init__(state_dim=2, noise_R_tril=noise_R_tril)

        valid_modalities = {"image", "pos", "sensors"}
        assert len(valid_modalities | modalities) == 3, "Received invalid modality"
        assert len(modalities) > 0, "Received empty modality list"
        self.modalities = modalities

        if "image" in modalities:
            self.observation_image_layers = layers.observation_image_layers(
                units, spanning_avg_pool=True
            )
        if "pos" in modalities:
            self.observation_pos_layers = layers.observation_pos_layers(units)
        if "sensors" in modalities:
            self.observation_sensors_layers = layers.observation_sensors_layers(units)

        self.shared_layers = nn.Sequential(
            nn.Linear(units * (len(modalities)), units * 2),
            nn.ReLU(inplace=True),
            resblocks.Linear(units * 2),
            resblocks.Linear(units * 2),
            # nn.Linear(units, units),
        )

        self.r_layer = nn.Sequential(
            nn.Linear(units, self.state_dim),
            nn.ReLU(inplace=True),
            resblocks.Linear(self.state_dim),
            nn.Linear(self.state_dim, self.state_dim),
        )

        self.z_layer = nn.Sequential(
            nn.Linear(units, self.state_dim),
            nn.ReLU(inplace=True),
            resblocks.Linear(self.state_dim),
            nn.Linear(self.state_dim, self.state_dim),
        )

        self.units = units
        self.add_R_noise = torch.ones(self.state_dim) * add_R_noise

    def forward(self, *, observations: types.ObservationsTorch) -> types.StatesTorch:
        assert type(observations) == dict
        observations = cast(types.TorchDict, observations)

        # N := distinct trajectory count
        # M := particle count
        N, _ = observations["gripper_pos"].shape

        # Construct observations feature vector
        # (N, obs_dim)
        obs = []
        if "image" in self.modalities:
            obs.append(
                self.observation_image_layers(observations["image"][:, None, :, :])
            )
        if "pos" in self.modalities:
            obs.append(self.observation_pos_layers(observations["gripper_pos"]))
        if "sensors" in self.modalities:
            obs.append(self.observation_sensors_layers(observations["gripper_sensors"]))
        observation_features = torch.cat(obs, dim=1)
        assert observation_features.shape == (N, self.units * len(obs))

        shared_features = self.shared_layers(observation_features)

        shared_features_z = shared_features[:, : self.units].clone()
        measurement_prediction = self.z_layer(shared_features_z)
        assert measurement_prediction.shape == (N, self.state_dim)

        if self.noise_R_tril is None:
            lt_hat = self.r_layer(shared_features[:, self.units :].clone())

        else:
            lt_hat = self.noise_R_tril

        lt = torch.diag_embed(lt_hat, offset=0, dim1=-2, dim2=-1)
        assert lt.shape == (N, self.state_dim, self.state_dim)

        measurement_covariance = lt ** 2
        if self.add_R_noise[0] > 0:
            measurement_covariance += torch.diag(self.add_R_noise).to(
                measurement_covariance.device
            )

        return measurement_prediction, measurement_covariance
