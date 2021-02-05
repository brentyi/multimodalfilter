from typing import List, cast

import fannypack
import numpy as np
import torch.utils.data
import torchfilter

# These need to externally set before training
buddy: fannypack.utils.Buddy
filter_model: torchfilter.base.Filter
trajectories: List[torchfilter.types.TrajectoryNumpy]
num_workers: int


def configure(
    *,
    buddy: fannypack.utils.Buddy,
    trajectories: List[torchfilter.types.TrajectoryNumpy],
    num_workers: int = 8,
):
    """Configure global settings for training helpers."""
    assert isinstance(buddy.model, torchfilter.base.Filter)
    globals()["buddy"] = buddy
    globals()["filter_model"] = cast(torchfilter.base.Filter, buddy.model)
    globals()["trajectories"] = trajectories
    globals()["num_workers"] = num_workers


# Training helpers
def train_pf_dynamics_single_step(*, epochs, batch_size=32, model=None):
    if model is None:
        model = filter_model
    assert isinstance(model, torchfilter.base.Filter)

    # Put model in train mode
    model.train()

    dataloader = torch.utils.data.DataLoader(
        torchfilter.data.SingleStepDataset(trajectories=trajectories),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    for _ in range(epochs):
        torchfilter.train.train_dynamics_single_step(
            buddy, model.dynamics_model, dataloader, loss_function="mse"
        )


def train_pf_dynamics_recurrent(
    *, subsequence_length, epochs, batch_size=32, model=None
):
    assert isinstance(filter_model, torchfilter.base.Filter)

    if model is None:
        model = filter_model
    assert isinstance(model, torchfilter.base.Filter)

    # Put model in train mode
    model.train()

    dataloader = torch.utils.data.DataLoader(
        torchfilter.data.SubsequenceDataset(
            trajectories=trajectories, subsequence_length=subsequence_length
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    for _ in range(epochs):
        torchfilter.train.train_dynamics_recurrent(
            buddy, model.dynamics_model, dataloader, loss_function="mse"
        )


def train_pf_measurement(*, epochs, batch_size, cov_scale=0.1):
    assert isinstance(filter_model, torchfilter.filters.ParticleFilter)

    # Put model in train mode
    filter_model.train()

    dataloader = torch.utils.data.DataLoader(
        torchfilter.data.ParticleFilterMeasurementDataset(
            trajectories=trajectories,
            covariance=np.identity(filter_model.state_dim) * cov_scale,
            samples_per_pair=10,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    for _ in range(epochs):
        torchfilter.train.train_particle_filter_measurement(
            buddy, filter_model.measurement_model, dataloader
        )


def train_kf_measurement(
    *, epochs, batch_size=32, model=None, optimizer_name="train_measurement"
):

    if model is None:
        model = filter_model
    assert isinstance(model, torchfilter.filters.VirtualSensorExtendedKalmanFilter)

    # Put model in train mode
    model.train()

    dataloader = torch.utils.data.DataLoader(
        torchfilter.data.SingleStepDataset(trajectories=trajectories),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    for _ in range(epochs):
        torchfilter.train.train_virtual_sensor(
            buddy,
            model.virtual_sensor_model,
            dataloader,
            optimizer_name=optimizer_name,
        )


def train_e2e(
    *,
    subsequence_length,
    epochs,
    batch_size=32,
    initial_cov_scale=0.1,
    measurement_initialize=False,
    model=None,
    optimizer_name="train_filter_recurrent",
):

    if model is None:
        model = filter_model
    assert isinstance(model, torchfilter.base.Filter)

    # Put model in train mode
    model.train()

    dataloader = torch.utils.data.DataLoader(
        torchfilter.data.SubsequenceDataset(
            trajectories=trajectories, subsequence_length=subsequence_length
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    initial_covariance = (
        torch.eye(model.state_dim, device=buddy.device) * initial_cov_scale
    )
    for _ in range(epochs):
        torchfilter.train.train_filter(
            buddy,
            model,
            dataloader,
            initial_covariance=initial_covariance,
            measurement_initialize=measurement_initialize,
            optimizer_name=optimizer_name,
        )
