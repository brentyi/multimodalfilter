import abc
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torchfilter
from torchfilter import types

from .utility import weighted_average


class CrossmodalKalmanFilterWeightModel(nn.Module, abc.ABC):
    """Crossmodal weight model."""

    def __init__(self, modality_count: int, state_dim: int):
        super().__init__()

        self.modality_count = modality_count
        self.state_dim = state_dim
        """int: Number of modalities."""

    @abc.abstractmethod
    def forward(self, *, observations: types.ObservationsTorch) -> torch.Tensor:
        """Compute log-modality weights.

        Args:
            observations (types.ObservationsTorch): Model observations.

        Returns:
            torch.Tensor: Computed weights for states. Shape should be
            `(modality_count, N, state_dim)`.
            torch.Tensor: Computed weights for state covariances. Shape should be
            `(modality_count, N, state_dim, state_dim)`.
        """
        pass


class CrossmodalKalmanFilter(torchfilter.base.Filter):
    """Utility class for merging unimodal kalman filter models via crossmodal weighting."""

    def __init__(
        self,
        *,
        filter_models: List[torchfilter.filters.VirtualSensorExtendedKalmanFilter],
        crossmodal_weight_model: CrossmodalKalmanFilterWeightModel,
        state_dim: int,
    ):
        super().__init__(state_dim=state_dim)

        self.filter_models = nn.ModuleList(filter_models)
        """ nn.ModuleList: List of measurement models. """

        self.crossmodal_weight_model = crossmodal_weight_model
        """ crossmodal.base_models.CrossmodalKalmanFilterWeightModel: Crossmodal
        weight model; should output one weight per measurement model. """

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
        """Kalman filter with crossmodal weights forward pass, single timestep.

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

        unimodal_states, unimodal_covariances = self.calculate_unimodal_states(
            observations, controls
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

        if np.sum(self._enabled_models) < len(self._enabled_models):
            state_weights = torch.from_numpy(
                np.array(self._enabled_models).astype(np.float32)
            )
            state_weights = (
                state_weights.unsqueeze(-1).unsqueeze(-1).repeat(1, N, self.state_dim)
            )
            state_weights = state_weights.to(unimodal_states.device)
        else:
            state_weights = self.crossmodal_weight_model(observations=observations)
        state_weights = state_weights[self._enabled_models]
        # note: my crossmodal weights will look different in output than PF
        assert state_weights.shape == (np.sum(self._enabled_models), N, self.state_dim)

        weighted_states, weighted_covariances = self.calculate_weighted_states(
            state_weights, unimodal_states, unimodal_covariances
        )

        assert weighted_states.shape == (N, self.state_dim)
        assert weighted_covariances.shape == (N, self.state_dim, self.state_dim)

        self.weighted_covariances = weighted_covariances

        for f in self.filter_models:
            f.states_prev = weighted_states
            f.states_covariance_prev = weighted_covariances

        return weighted_states

    def calculate_weighted_states(
        self, state_weights, unimodal_states, unimodal_covariances
    ):
        model_dim, N, state_dim = state_weights.shape
        assert model_dim == np.sum(self._enabled_models)
        assert state_dim == self.state_dim

        weighted_states = weighted_average(unimodal_states, state_weights)
        covariance_weights = state_weights.unsqueeze(-1).repeat(
            (1, 1, 1, self.state_dim)
        )
        covariance_weights = covariance_weights * covariance_weights.transpose(-1, -2)
        weighted_covariances = torch.sum(covariance_weights * unimodal_covariances, 0)

        return weighted_states, weighted_covariances

    def calculate_unimodal_states(self, observations, controls):
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

        return unimodal_states, unimodal_covariances

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

    def measurement_initialize_beliefs(self, observations):
        N = observations[[*observations][0]].shape[0]

        model_list = [
            filter_model.virtual_sensor_model(observations=observations)
            for i, filter_model in enumerate(self.filter_models)
            if self._enabled_models[i]
        ]

        unimodal_states = torch.stack([x[0] for x in model_list])
        unimodal_scale_trils = torch.stack([x[1] for x in model_list])
        unimodal_covariances = unimodal_scale_trils @ unimodal_scale_trils.transpose(
            -1, -2
        )

        state_weights = self.crossmodal_weight_model(observations=observations)
        state_weights = state_weights[self._enabled_models]

        weighted_states = weighted_average(unimodal_states, state_weights)
        covariance_multiplier = (
            torch.prod(torch.prod(state_weights, dim=-1), dim=0)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        assert covariance_multiplier.shape == (N, 1, 1)
        weighted_covariances = covariance_multiplier * torch.sum(
            unimodal_covariances, dim=0
        )

        assert weighted_states.shape == (N, self.state_dim)
        assert weighted_covariances.shape == (N, self.state_dim, self.state_dim)

        self.initialize_beliefs(mean=weighted_states, covariance=weighted_covariances)


class CrossmodalVirtualSensorModel(torchfilter.base.VirtualSensorModel):
    """Utility class for merging unimodal measurement models via crossmodal weighting."""

    def __init__(
        self,
        *,
        virtual_sensor_model: List[torchfilter.base.VirtualSensorModel],
        crossmodal_weight_model: CrossmodalKalmanFilterWeightModel,
        state_dim: int,
    ):
        super().__init__(state_dim=state_dim)

        self.virtual_sensor_model = nn.ModuleList(virtual_sensor_model)
        """ nn.ModuleList: List of measurement models. """

        self.crossmodal_weight_model = crossmodal_weight_model
        """ crossmodal.base_models.CrossmodalKalmanFilterWeightModel: Crossmodal
        weight model; should output one weight per measurement model. """

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

    def forward(
        self, *, observations: types.ObservationsTorch
    ) -> Tuple[types.StatesTorch, types.ScaleTrilTorch]:
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

        if np.sum(self._enabled_models) < len(self._enabled_models):
            state_weights = torch.from_numpy(
                np.array(self._enabled_models).astype(np.float32)
            )
            state_weights = (
                state_weights.unsqueeze(-1).unsqueeze(-1).repeat(1, N, self.state_dim)
            )
            state_weights = state_weights.to(unimodal_states.device)
        else:
            state_weights = self.crossmodal_weight_model(observations=observations)
        state_weights = state_weights[self._enabled_models]

        # note: my crossmodal weights will look different in output than PF
        assert state_weights.shape == (np.sum(self._enabled_models), N, self.state_dim)

        weighted_states = weighted_average(unimodal_states, state_weights)
        covariance_multiplier = (
            torch.prod(torch.prod(state_weights, dim=-1), dim=0)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        assert covariance_multiplier.shape == (N, 1, 1)
        weighted_covariances = covariance_multiplier * torch.sum(
            unimodal_covariances, dim=0
        )

        assert weighted_states.shape == (N, self.state_dim)
        assert weighted_covariances.shape == (N, self.state_dim, self.state_dim)

        return weighted_states, torch.cholesky(weighted_covariances)
