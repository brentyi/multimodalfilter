from typing import List, cast

import numpy as np
import torch.utils.data

import diffbayes
import fannypack

# These need to externally set before training
buddy: fannypack.utils.Buddy
filter_model: diffbayes.base.Filter
trajectories: List[diffbayes.types.TrajectoryTupleNumpy]
num_workers: int


def configure(
    *,
    buddy: fannypack.utils.Buddy,
    trajectories: List[diffbayes.types.TrajectoryTupleNumpy],
    num_workers: int = 8,
):
    """Configure global settings for training helpers.
    """
    assert isinstance(buddy.model, diffbayes.base.Filter)
    globals()["buddy"] = buddy
    globals()["filter_model"] = cast(diffbayes.base.Filter, buddy.model)
    globals()["trajectories"] = trajectories
    globals()["num_workers"] = num_workers


# Training helpers
def train_pf_dynamics_single_step(*, epochs, batch_size=32, model=None):
    if model is None:
        model = filter_model
    assert isinstance(model, diffbayes.base.Filter)

    # Put model in train mode
    model.train()

    dataloader = torch.utils.data.DataLoader(
        diffbayes.data.SingleStepDataset(trajectories=trajectories),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    for _ in range(epochs):
        diffbayes.train.train_dynamics_single_step(
            buddy, model.dynamics_model, dataloader, loss_function="mse"
        )


def train_pf_dynamics_recurrent(*, subsequence_length, epochs, batch_size=32, model=None):
    assert isinstance(filter_model, diffbayes.base.Filter)

    if model is None:
        model = filter_model
    assert isinstance(model, diffbayes.base.Filter)

    # Put model in train mode
    model.train()

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
            buddy, model.dynamics_model, dataloader, loss_function="mse"
        )


def train_pf_measurement(*, epochs, batch_size, cov_scale=0.1):
    assert isinstance(filter_model, diffbayes.base.ParticleFilter)

    # Put model in train mode
    filter_model.train()

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

def train_kf_measurement(*, epochs, batch_size=32, model=None):

    if model is None:
        model = filter_model
    assert isinstance(model, diffbayes.base.KalmanFilter)

    # Put model in train mode
    model.train()

    dataloader = torch.utils.data.DataLoader(
        diffbayes.data.SingleStepDataset(trajectories=trajectories),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    for _ in range(epochs):
        diffbayes.train.train_kalman_filter_measurement_model(
            buddy, model.measurement_model, dataloader, loss_function="mse"
        )

def train_e2e(*, subsequence_length, epochs, batch_size=32, initial_cov_scale=0.1, measurement_init=False, model=None):

    if model is None:
        model = filter_model
    assert isinstance(model, diffbayes.base.Filter)

    # Put model in train mode
    model.train()

    dataloader = torch.utils.data.DataLoader(
        diffbayes.data.SubsequenceDataset(
            trajectories=trajectories, subsequence_length=subsequence_length
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    initial_covariance = (
        torch.eye(model.state_dim, device=buddy.device) * initial_cov_scale
    )
    for _ in range(epochs):
        diffbayes.train.train_filter(
            buddy, model, dataloader, initial_covariance=initial_covariance,
            measurement_initialization=measurement_init,
        )
