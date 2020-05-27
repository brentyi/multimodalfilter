from typing import List, Optional

import numpy as np
import torch.utils.data

import diffbayes
import fannypack

# These need to externally set before running
buddy: fannypack.utils.Buddy = None
filter_model: diffbayes.base.Filter = None
trajectories: List[diffbayes.types.TrajectoryTupleNumpy] = None
num_workers: int = None


def configure(
    buddy: fannypack.utils.Buddy,
    filter_model: diffbayes.base.Filter,
    trajectories: List[diffbayes.types.TrajectoryTupleNumpy],
    num_workers: int = 8,
):
    """Configure global settings for training helpers.
    """
    globals()["buddy"] = buddy
    globals()["filter_model"] = filter_model
    globals()["trajectories"] = trajectories
    globals()["num_workers"] = num_workers


# Training helpers
def train_dynamics_single_step(*, epochs, batch_size=32):
    assert isinstance(filter_model, diffbayes.base.ParticleFilter)

    dataloader = torch.utils.data.DataLoader(
        diffbayes.data.SingleStepDataset(trajectories=trajectories),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    for _ in range(epochs):
        diffbayes.train.train_dynamics_single_step(
            buddy, filter_model.dynamics_model, dataloader
        )


def train_dynamics_recurrent(*, subsequence_length, epochs, batch_size=32):
    assert isinstance(filter_model, diffbayes.base.ParticleFilter)

    dataloader = torch.utils.data.DataLoader(
        diffbayes.data.SubsequenceDataset(
            trajectories=trajectories, subsequence_length=subsequence_length
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    for _ in range(epochs):
        diffbayes.train.train_dynamics_recurrent(
            buddy, filter_model.dynamics_model, dataloader
        )


def train_pf_measurement(*, epochs, batch_size, cov_scale=0.1):
    assert isinstance(filter_model, diffbayes.base.ParticleFilter)
    dataloader = torch.utils.data.DataLoader(
        diffbayes.data.ParticleFilterMeasurementDataset(
            trajectories=trajectories,
            covariance=np.identity(filter_model.state_dim) * cov_scale,
            samples_per_pair=10,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    for _ in range(epochs):
        diffbayes.train.train_particle_filter_measurement_model(
            buddy, filter_model.measurement_model, dataloader
        )


def train_e2e(*, subsequence_length, epochs, batch_size=32, initial_cov_scale=0.1):
    assert isinstance(filter_model, diffbayes.base.Filter)
    dataloader = torch.utils.data.DataLoader(
        diffbayes.data.SubsequenceDataset(
            trajectories=trajectories, subsequence_length=subsequence_length
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    initial_covariance = (
        torch.eye(filter_model.state_dim, device=buddy.device) * initial_cov_scale
    )
    for _ in range(epochs):
        diffbayes.train.train_filter(
            buddy, filter_model, dataloader, initial_covariance=initial_covariance
        )
