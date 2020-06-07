import dataclasses
from typing import Dict, List

import numpy as np
import torch

import diffbayes
import fannypack


@dataclasses.dataclass(frozen=True)
class EvalResults:
    raw_rmse: List[float]
    theta_rmse_deg: float
    x_rmse_cm: float
    y_rmse_cm: float


def eval_filter(
    filter_model: diffbayes.base.Filter,
    trajectories: List[diffbayes.types.TrajectoryTupleNumpy],
) -> EvalResults:
    """Evaluate a filter and print out metrics.

    Args:
        filter_model (diffbayes.base.Filter): Filter to evaluate.
        trajectories (list): List of trajectories to evaluate on; typically outputted
            from `crossmodal.door_data.load_trajectories()`.
    """
    assert isinstance(filter_model, diffbayes.base.Filter)
    assert type(trajectories) == list
    assert not filter_model.training, "Model needs to be in eval mode!"

    # Convert list of trajectories -> batch
    #
    # Note that we need to jump through some hoops because the observations are stored as
    # dictionaries
    states = []
    observations = fannypack.utils.SliceWrapper({})
    controls = []
    min_timesteps = min([s.shape[0] for s, o, c in trajectories])
    for (s, o, c) in trajectories:
        # Update batch
        states.append(s[:min_timesteps])
        observations.append(fannypack.utils.SliceWrapper(o)[:min_timesteps])
        controls.append(c[:min_timesteps])

    # Numpy -> torch
    # > We create a batch axis @ index 1
    device = next(filter_model.parameters()).device
    stack_fn = lambda list_value: fannypack.utils.to_torch(
        np.stack(list_value, axis=1), device=device
    )

    states = stack_fn(states)
    observations = observations.map(stack_fn)
    controls = stack_fn(controls)

    # Get sequence length (T) and batch size (N) dimensions
    assert states.shape[:2] == controls.shape[:2]
    assert states.shape[:2] == fannypack.utils.SliceWrapper(observations).shape[:2]
    T, N = states.shape[:2]

    # Initialize beliefs
    state_dim = filter_model.state_dim
    cov = (torch.eye(state_dim, device=device) * 0.1)[None, :, :].expand(
        (N, state_dim, state_dim)
    )
    filter_model.initialize_beliefs(
        mean=states[0], covariance=cov,
    )

    # Run filter
    with torch.no_grad():
        predicted_states = filter_model.forward_loop(
            observations=fannypack.utils.SliceWrapper(observations)[1:],
            controls=controls[1:],
        )

    # Validate predicted states
    T = predicted_states.shape[0]
    assert predicted_states.shape == (T, N, state_dim)

    # Compute & update errors
    true_states = states[1:]
    start_truncation = 30
    mse = np.mean(
        fannypack.utils.to_numpy(
            predicted_states[start_truncation:] - true_states[start_truncation:]
        ).reshape((-1, state_dim))
        ** 2,
        axis=0,
    )
    raw_rmse = np.sqrt(mse / len(trajectories))
    rmse = raw_rmse * np.array([0.39479038, 0.05650279, 0.0565098])
    results = EvalResults(
        raw_rmse=list(raw_rmse),
        theta_rmse_deg=rmse[0] * 180.0 / np.pi,
        x_rmse_cm=rmse[1] * 100.0,
        y_rmse_cm=rmse[2] * 100.0,
    )
    print()
    print("-----")
    print(f"Raw RMSE:   {results.raw_rmse}")
    print("-----")
    print(f"Theta RMSE: {results.theta_rmse_deg:.8f} degrees")
    print(f"X RMSE:     {results.x_rmse_cm:.8f} cm")
    print(f"Y RMSE:     {results.y_rmse_cm:.8f} cm")
    print("-----")

    return results
