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


def train_kf_measurement(*, epochs, batch_size=32, model=None,
                         optimizer_name="train_measurement"):

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
            buddy, model.measurement_model, dataloader,
            optimizer_name=optimizer_name,
        )


def train_e2e(*, subsequence_length, epochs, batch_size=32, initial_cov_scale=0.1,
              measurement_initialize=False, model=None,
              optimizer_name="train_filter_recurrent"):

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
        drop_last=True,
    )
    initial_covariance = (
        torch.eye(model.state_dim, device=buddy.device) * initial_cov_scale
    )
    for _ in range(epochs):
        diffbayes.train.train_filter(
            buddy, model, dataloader, initial_covariance=initial_covariance,
            measurement_initialize=measurement_initialize,
            optimizer_name=optimizer_name
        )



"""
BELOW IS PROTOTYPING CODE FOR TRAINING CROSSMODAL EKFs

"""

import torch.nn.functional as F
from tqdm.auto import tqdm

def _swap_batch_sequence_axes(tensor):
    """Converts data formatted as (N, T, ...) to (T, N, ...)
    """
    return torch.transpose(tensor, 0, 1)


def train_crossmodal_filter(
    buddy: fannypack.utils.Buddy,
    filter_model: diffbayes.base.Filter,
    dataloader,
    initial_covariance: torch.Tensor,
    *,
    loss_function= F.mse_loss,
    log_interval: int = 10,
    measurement_initialize=False,
    optimizer_name="train_filter_recurrent",
) -> None:
    """Trains a filter end-to-end via backpropagation through time for 1 epoch over a
    subsequence dataset.

    """
    # Dataloader should load a SubsequenceDataset
    assert isinstance(dataloader.dataset, diffbayes.data.SubsequenceDataset)
    assert initial_covariance.shape == (filter_model.state_dim, filter_model.state_dim)
    assert filter_model.training, "Model needs to be set to train mode"

    # Track mean epoch loss
    epoch_loss = 0.0

    # Train filter model for 1 epoch
    with buddy.log_scope("train_filter_recurrent"):
        for batch_idx, batch_data in enumerate(tqdm(dataloader)):
            # Move data
            batch_gpu = fannypack.utils.to_device(batch_data, buddy.device)
            true_states, observations, controls = batch_gpu

            # Swap batch size, sequence length axes
            true_states = _swap_batch_sequence_axes(true_states)
            observations = fannypack.utils.SliceWrapper(observations).map(
                _swap_batch_sequence_axes
            )
            controls = fannypack.utils.SliceWrapper(controls).map(
                _swap_batch_sequence_axes
            )

            # Shape checks
            T, N, state_dim = true_states.shape
            assert state_dim == filter_model.state_dim
            assert fannypack.utils.SliceWrapper(observations).shape[:2] == (T, N)
            assert fannypack.utils.SliceWrapper(controls).shape[:2] == (T, N)
            assert batch_idx != 0 or N == dataloader.batch_size

            # Populate initial filter belief
            if measurement_initialize and hasattr(filter_model, 'measurement_initialize_beliefs'):
                filter_model.measurement_initialize_beliefs(fannypack.utils.SliceWrapper(observations)[0])
            else:
                initial_states_covariance = initial_covariance[None, :, :].expand(
                    (N, state_dim, state_dim)
                )
                scale_tril = torch.sqrt(initial_states_covariance)
                initial_states = torch.distributions.MultivariateNormal(
                    true_states[0], scale_tril=scale_tril,
                ).sample()
                filter_model.initialize_beliefs(
                    mean=initial_states, covariance=initial_states_covariance
                )

            def train_loop(obs, c, cm_filter):
                obs = fannypack.utils.SliceWrapper(obs)
                c = fannypack.utils.SliceWrapper(c)

                # Get sequence length (T), batch size (N)
                T, N = c.shape[:2]
                assert obs.shape[:2] == (T, N)

                # Filtering forward pass
                # We treat t = 0 as a special case to make it easier to create state_predictions
                # tensor on the correct device
                t = 0
                current_prediction = cm_filter(observations=obs[t], controls=c[t])
                state_pred = current_prediction.new_zeros((T, N, cm_filter.state_dim))
                state_pred[t] = current_prediction

                # todo: we assume here there are two modalities
                weight_pred = current_prediction.new_zeros((T, 2, N, cm_filter.state_dim))
                weight_pred[t] = cm_filter.crossmodal_weight_model(observations=obs[t])

                unimodal_pred = current_prediction.new_zeros((T, 2, N, cm_filter.state_dim))
                unimodal_pred[t] = cm_filter.unimodal_states

                for t in range(1, T):
                    # Compute state prediction for a single timestep
                    # We use __call__ to make sure hooks are dispatched correctly
                    current_prediction = cm_filter(observations=obs[t], controls=c[t])

                    # Validate & add to output
                    state_pred[t] = current_prediction
                    weight_pred[t] = cm_filter.crossmodal_weight_model(observations=obs[t])
                    unimodal_pred[t] = cm_filter.unimodal_states

                return state_pred, weight_pred, unimodal_pred

            state_predictions, weight_predictions, unimodal_pred = train_loop(
                fannypack.utils.SliceWrapper(observations)[1:],
                fannypack.utils.SliceWrapper(controls)[1:],
                filter_model,
            )

            # Minimize loss
            loss_crossmodal = loss_function(state_predictions, true_states[1:])
            loss_unimodal = loss_function(unimodal_pred[:, 0], true_states[1:]) +  \
                            loss_function(unimodal_pred[:, 1], true_states[1:])

            loss = 0.8 * loss_crossmodal + 0.2 * loss_unimodal
            buddy.minimize(loss, optimizer_name=optimizer_name)
            epoch_loss += fannypack.utils.to_numpy(loss)

            # Logging
            if batch_idx % log_interval == 0:
                buddy.log("loss", loss)
                buddy.log("loss_unimodal", loss_unimodal)
                buddy.log("loss_crossmodal", loss_crossmodal)
                buddy.log_gradient_norm()
                buddy.log("Pred  mean", state_predictions.mean())
                buddy.log("Pred  std", state_predictions.std())
                buddy.log("Weight 0 Pred mean", weight_predictions[:, 0].mean())
                buddy.log("Weight 0 Pred std",  weight_predictions[:, 0].std())
                buddy.log("Weight 1 Pred mean", weight_predictions[:, 1].mean())
                buddy.log("Weight 1 Pred std",  weight_predictions[:, 1].std())

    # Print average training loss
    epoch_loss /= len(dataloader)
    print("(train_filter) Epoch training loss: ", epoch_loss)


def train_cm_e2e(*, subsequence_length, epochs, batch_size=32, initial_cov_scale=0.1,
              measurement_initialize=False, model=None,
              optimizer_name="train_filter_recurrent"):

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
        drop_last=True,
    )
    initial_covariance = (
        torch.eye(model.state_dim, device=buddy.device) * initial_cov_scale
    )
    for _ in range(epochs):
        train_crossmodal_filter(
            buddy, model, dataloader,
            initial_covariance=initial_covariance,
            measurement_initialize=measurement_initialize,
            optimizer_name=optimizer_name
        )