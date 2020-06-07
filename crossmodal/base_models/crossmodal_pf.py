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
    diffbayes.base.ParticleFilterMeasurementModel
):
    """Utility class for merging unimodal measurement models via crossmodal weighting.
    """

    def __init__(
        self,
        *,
        measurement_models: List[diffbayes.base.ParticleFilterMeasurementModel],
        crossmodal_weight_model: CrossmodalWeightModel,
    ):
        self.measurement_models = nn.ModuleList(measurement_models)
        """ nn.ModuleList: List of measurement models. """

        self.crossmodal_weight_model = crossmodal_weight_model
        """ crossmodal.base_models.CrossmodalParticleFilterWeightModel: Crossmodal
        weight model; should output one weight per measurement model. """

        self.enabled_models: List[bool] = [True for _ in self.measurement_models]
        """ List[bool]: List of booleans, one for each measurement model: set flag to
        False to disable a modality."""

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
                measurement_model(states, observations)
                for i, measurement_model in enumerate(self.measurement_models)
                if self.enabled_models[i]
            ],
            dim=2,
        )
        assert unimodal_log_likelihoods.shape == (N, M, np.sum(self.enabled_models))

        # Compute modality-specific weights
        modality_log_weights = self.crossmodal_weight_model(observations)[
            :, :, self.enabled_models
        ]
        assert modality_log_weights.shape == (N, M, np.sum(self.enabled_models))

        # Weight particle likelihoods by modality & return
        particle_log_likelihoods = torch.logsumexp(
            modality_log_weights + unimodal_log_likelihoods, dim=2
        )
        assert particle_log_likelihoods == (N, M)

        return particle_log_likelihoods
