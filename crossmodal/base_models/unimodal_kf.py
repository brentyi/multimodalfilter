import abc
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torchfilter
from torchfilter import types

from .utility import weighted_average


class UnimodalVirtualSensorModel(torchfilter.base.VirtualSensorModel):
    """Utility class for merging unimodal measurement models via unimodal weighting."""

    def __init__(
        self,
        *,
        virtual_sensor_model: List[torchfilter.base.VirtualSensorModel],
        state_dim: int,
    ):
        super().__init__(state_dim=state_dim)

        self.virtual_sensor_model = nn.ModuleList(virtual_sensor_model)
        """ nn.ModuleList: List of measurement models. """

        self._enabled_models: List[bool] = [True for _ in self.virtual_sensor_model]

    @property
    def enabled_models(self) -> List[bool]:
        """List of enabled unimodal measurement models.

        Returns:
            List[bool]: List of booleans, one for each measurement model: set flag to
            False to disable a modality.
        """
        return self._enabled_models

    @enabled_models.setter
    def enabled_models(self, enabled_models: List[bool]) -> None:
        """Setter for the `enabled_models` property.

        Args:
            enabled_models (List[bool]): New value.
        """

        # Input validation
        assert isinstance(enabled_models, list)
        assert len(enabled_models) == len(self.virtual_sensor_model)
        for x in enabled_models:
            assert type(x) == bool

        # Assign value
        self._enabled_models = enabled_models

    def forward(self, *, observations: types.ObservationsTorch) -> types.StatesTorch:
        """Observation model forward pass, over batch size `N`.
        For each member of a batch, we expect one unique observation.

        Args:
            observations (dict or torch.Tensor): Measurement inputs. Should be
                either a dict of tensors or tensor of size `(N, ...)`.
        Returns:
            torch.Tensor: Measurement state prediction. Shape should be `(N, state_dim)`
            torch.Tensor: Measurement state prediction covariance. Shape should be `(N, state_dim, state_dim)`.
        """
        N = observations[[*observations][0]].shape[0]

        model_list = [
            (virtual_sensor_model(observations=observations))
            for i, virtual_sensor_model in enumerate(self.virtual_sensor_model)
            if self._enabled_models[i]
        ]

        unimodal_states = torch.stack([x[0] for x in model_list])
        unimodal_scale_trils = torch.stack([x[1] for x in model_list])
        unimodal_covariances = unimodal_scale_trils @ unimodal_scale_trils.transpose(
            -1, -2
        )

        if np.sum(self._enabled_models) == 1:
            weighted_states = unimodal_states[0]
            weighted_covariances = unimodal_covariances[0]
        else:
            unimodal_precision = torch.stack([1.0 / (x[1] + 1e-9) for x in model_list])

            assert unimodal_states.shape == (
                np.sum(self._enabled_models),
                N,
                self.state_dim,
            )
            assert unimodal_covariances.shape == (
                np.sum(self._enabled_models),
                N,
                self.state_dim,
                self.state_dim,
            )

            unimodal_weights = torch.diagonal(
                unimodal_precision, dim1=-2, dim2=-1
            ).squeeze(1)
            assert unimodal_weights.shape == unimodal_states.shape

            weighted_states = weighted_average(unimodal_states, unimodal_weights)
            weighted_covariances = torch.inverse(
                torch.sum(unimodal_precision, dim=0) + 1e-9
            )

        assert weighted_states.shape == (
            N,
            self.state_dim,
        )
        assert weighted_covariances.shape == (N, self.state_dim, self.state_dim)

        return weighted_states, weighted_covariances


class UnimodalKalmanFilter(torchfilter.base.Filter):
    """Utility class for merging unimodal kalman filter models via crossmodal weighting."""

    def __init__(
        self,
        *,
        filter_models: List[torchfilter.filters.VirtualSensorExtendedKalmanFilter],
        state_dim: int,
    ):
        super().__init__(state_dim=state_dim)

        self.filter_models = nn.ModuleList(filter_models)
        """ nn.ModuleList: List of measurement models. """

        self._enabled_models: List[bool] = [True for _ in self.filter_models]
        self.weighted_covariances = None

    @property
    def enabled_models(self) -> List[bool]:
        """List of enabled unimodal measurement models.

        Returns:
            List[bool]: List of booleans, one for each measurement model: set flag to
            False to disable a modality.
        """
        return self._enabled_models

    @enabled_models.setter
    def enabled_models(self, enabled_models: List[bool]) -> None:
        """Setter for the `enabled_models` property.

        Args:
            enabled_models (List[bool]): New value.
        """

        # Input validation
        assert isinstance(enabled_models, list)
        assert len(enabled_models) == len(self.filter_models)
        for x in enabled_models:
            assert type(x) == bool

        # Assign value
        self._enabled_models = enabled_models

    def forward(
        self,
        *,
        observations: types.ObservationsTorch,
        controls: types.ControlsTorch,
    ) -> types.StatesTorch:
        """Kalman filter with unimodal weights forward pass, single timestep.

        Args:
            observations (dict or torch.Tensor): Measurement inputs. Should be
                either a dict of tensors or tensor of size `(N, ...)`.
            controls (dict or torch.Tensor): control inputs. should be either a
                dict of tensors or tensor of shape `(N, ...)`.
        Returns:
            torch.Tensor: Weighted filter state estimation. Shape should be `(N, state_dim)`
            torch.Tensor: Weighted filter state estimation covariance. Shape should be `(N, state_dim, state_dim)`.
        """

        N, _ = controls.shape

        unimodal_states = torch.stack(
            [
                (filter_model(observations=observations, controls=controls))
                for i, filter_model in enumerate(self.filter_models)
                if self._enabled_models[i]
            ]
        )

        unimodal_covariances = torch.stack(
            [
                filter_model.state_covariance_estimate
                for i, filter_model in enumerate(self.filter_models)
                if self._enabled_models[i]
            ]
        )

        if np.sum(self._enabled_models) == 1:
            weighted_states = unimodal_states[0]
            weighted_covariances = unimodal_covariances[0]

        else:

            unimodal_precision = torch.stack(
                [
                    torch.inverse(filter_model.state_covariance_estimate + 1e-9)
                    for i, filter_model in enumerate(self.filter_models)
                    if self._enabled_models[i]
                ]
            )

            assert unimodal_states.shape == (
                np.sum(self._enabled_models),
                N,
                self.state_dim,
            )
            assert unimodal_covariances.shape == (
                np.sum(self._enabled_models),
                N,
                self.state_dim,
                self.state_dim,
            )

            weighted_covariances = torch.inverse(
                torch.sum(unimodal_precision, dim=0) + 1e-9
            )

            unimodal_states_reshaped = unimodal_states.reshape(-1, self.state_dim, 1)
            unimodal_precision_reshaped = unimodal_precision.reshape(
                -1, self.state_dim, self.state_dim
            )

            weighted_states_unnormalized = torch.sum(
                unimodal_precision_reshaped.bmm(unimodal_states_reshaped).reshape(
                    np.sum(self._enabled_models), N, self.state_dim, 1
                ),
                dim=0,
            )

            weighted_states = weighted_covariances.bmm(
                weighted_states_unnormalized
            ).squeeze(-1)

        assert weighted_states.shape == (
            N,
            self.state_dim,
        )
        assert weighted_covariances.shape == (N, self.state_dim, self.state_dim)

        return weighted_states

    @property
    def state_covariance_estimate(self):
        return self.weighted_covariances

    def initialize_beliefs(self, *, mean: torch.Tensor, covariance: torch.Tensor):
        """Set kalman state prediction and state covariance to mean and covariance.

        Args:
            mean (torch.Tensor): Mean of belief. Shape should be
                `(N, state_dim)`.
            covariance (torch.Tensor): Covariance of belief. Shape should be
                `(N, state_dim, state_dim)`.
        """
        N = mean.shape[0]
        assert mean.shape == (N, self.state_dim)
        assert covariance.shape == (N, self.state_dim, self.state_dim)

        for model in self.filter_models:
            model.initialize_beliefs(mean=mean, covariance=covariance)
