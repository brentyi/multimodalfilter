import abc
from typing import List

import numpy as np
import torch
import torch.nn as nn

import diffbayes
from diffbayes import types


class CrossmodalWeightModel(nn.Module, abc.ABC):
    """Crossmodal weight model.
    """

    def __init__(self, modality_count: int):
        super().__init__()

        self.modality_count = modality_count
        """int: Number of modalities."""

    @abc.abstractmethod
    def forward(self, *, observations: types.ObservationsTorch) -> torch.Tensor:
        """Compute log-modality weights.

        Args:
            observations (types.ObservationsTorch): Model observations.

        Returns:
            torch.Tensor: Computed weights. Shape should be
            `(N, modality_count, state_dim +
            (state_dim)(state_dim-1)/2`.
        """
        pass


class CrossmodalKalmanFilterMeasurementModel(
    diffbayes.base.KalmanFilterMeasurementModel
):
    """Utility class for merging unimodal measurement models via crossmodal weighting.
    """

    def __init__(
        self,
        *,
        measurement_models: List[diffbayes.base.KalmanFilterMeasurementModel],
        crossmodal_weight_model: CrossmodalWeightModel,
        state_dim: int,
    ):
        super().__init__(state_dim=state_dim)

        self.measurement_models = nn.ModuleList(measurement_models)
        """ nn.ModuleList: List of measurement models. """

        self.crossmodal_weight_model = crossmodal_weight_model
        """ crossmodal.base_models.CrossmodalKalmanFilterWeightModel: Crossmodal
        weight model; should output one weight per measurement model. """

        self._enabled_models: List[bool] = [True for _ in self.measurement_models]

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
        assert len(enabled_models) == len(self.measurement_models)
        for x in enabled_models:
            assert type(x) == bool

        # Assign value
        self._enabled_models = enabled_models

    def forward(
        self, *, observations: types.ObservationsTorch
    ) -> types.StatesTorch:
        """Observation model forward pass, over batch size `N`.
        For each member of a batch, we expect `M` separate states (particles)
        and one unique observation.

        Args:
            observations (dict or torch.Tensor): Measurement inputs. Should be
                either a dict of tensors or tensor of size `(N, ...)`.
        Returns:
            torch.Tensor: Measurement state prediction. Shape should be `(N, state_dim)`
            torch.Tensor: Measurement state prediction covariance. Shape should be `(N, state_dim, state_dim)`.
        """

        pass


class CrossmodalKalmanFilter(
    diffbayes.base.KalmanFilter
):
    """Utility class for merging unimodal kalman filter models via crossmodal weighting.
    """

    def __init__(
            self,
            *,
            filter_models: List[diffbayes.base.KalmanFilter],
            crossmodal_weight_model: CrossmodalWeightModel,
            state_dim: int,
    ):
        super().__init__(state_dim=state_dim)

        self.filter_models = nn.ModuleList(filter_models)
        """ nn.ModuleList: List of measurement models. """

        self.crossmodal_weight_model = crossmodal_weight_model
        """ crossmodal.base_models.CrossmodalKalmanFilterWeightModel: Crossmodal
        weight model; should output one weight per measurement model. """

        self._enabled_models: List[bool] = [True for _ in self.filter_models]

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
            self, *, observations: types.ObservationsTorch,
            controls: types.ControlsTorch,
    ) -> types.StatesTorch:
        """Kalman filter with crossmodal weights forward pass, single timestep.

        Args:
            observations (dict or torch.Tensor): Measurement inputs. Should be
                either a dict of tensors or tensor of size `(N, ...)`.
            controls (dict or torch.Tensor): control inputs. should be either a
                dict of tensors or tensor of shape `(N, ...)`.
        Returns:
            torch.Tensor: Measurement state prediction. Shape should be `(N, state_dim)`
            torch.Tensor: Measurement state prediction covariance. Shape should be `(N, state_dim, state_dim)`.
        """

        pass