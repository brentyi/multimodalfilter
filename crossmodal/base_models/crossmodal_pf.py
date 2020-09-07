import abc
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torchfilter
from torchfilter import types


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
            torch.Tensor: Computed weights. Shape should be `(N, modality_count)`.
        """
        pass


class CrossmodalParticleFilterMeasurementModel(
    torchfilter.base.ParticleFilterMeasurementModel
):
    """Utility class for merging unimodal measurement models via crossmodal weighting.

    If no crossmodal weight model is specified, we weight all measurement model outputs
    equally.
    """

    def __init__(
        self,
        *,
        measurement_models: List[torchfilter.base.ParticleFilterMeasurementModel],
        crossmodal_weight_model: Optional[CrossmodalWeightModel],
        state_dim: int,
    ):
        super().__init__(state_dim=state_dim)

        self.measurement_models = nn.ModuleList(measurement_models)
        """ nn.ModuleList: List of measurement models. """

        self.crossmodal_weight_model = crossmodal_weight_model
        """ Optional[crossmodal.base_models.CrossmodalParticleFilterWeightModel]:
        Crossmodal weight model; should output one weight per measurement model. """

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
        self, *, states: types.StatesTorch, observations: types.ObservationsTorch
    ) -> types.StatesTorch:
        """Observation model forward pass, over batch size `N`.
        For each member of a batch, we expect `M` separate states (particles)
        and one unique observation.

        Args:
            states (torch.Tensor): States to pass to our observation model.
                Shape should be `(N, M, state_dim)`.
            observations (dict or torch.Tensor): Measurement inputs. Should be
                either a dict of tensors or tensor of size `(N, ...)`.
        Returns:
            torch.Tensor: Log-likelihoods of each state, conditioned on a
            corresponding observation. Shape should be `(N, M)`.
        """
        N, M, _state_dim = states.shape

        # Unimodal particle likelihoods
        unimodal_log_likelihoods = torch.stack(
            [
                measurement_model(states=states, observations=observations)
                for i, measurement_model in enumerate(self.measurement_models)
                if self._enabled_models[i]
            ],
            dim=2,
        )
        assert unimodal_log_likelihoods.shape == (N, M, np.sum(self._enabled_models))

        # Compute modality-specific weights
        if self.crossmodal_weight_model is not None:
            modality_log_weights = self.crossmodal_weight_model(
                observations=observations
            )[:, self._enabled_models]
            assert modality_log_weights.shape == (N, np.sum(self._enabled_models))

            # Normalize each modality
            unimodal_log_likelihoods_norm = torch.zeros_like(unimodal_log_likelihoods)
            for i in range(unimodal_log_likelihoods.shape[2]):
                source = unimodal_log_likelihoods[:, :, i]
                unimodal_log_likelihoods_norm[:, :, i] = (
                    source - torch.max(source, dim=1, keepdim=True)[0]
                )

            # Weight particle likelihoods by modality & return
            particle_log_likelihoods = torch.logsumexp(
                modality_log_weights[:, None, :] + unimodal_log_likelihoods, dim=2
            )
            assert particle_log_likelihoods.shape == (N, M)
        else:
            # Weight particle likelihoods by modality & return
            particle_log_likelihoods = torch.logsumexp(unimodal_log_likelihoods, dim=2)
            assert particle_log_likelihoods.shape == (N, M)

        return particle_log_likelihoods
