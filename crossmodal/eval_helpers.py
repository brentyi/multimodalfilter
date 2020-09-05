from typing import Dict, List, Type, cast

import diffbayes
import fannypack
import numpy as np
import torch

from . import tasks

# These need to externally set before eval
buddy: fannypack.utils.Buddy
filter_model: diffbayes.base.Filter
task: Type[tasks.Task]
dataset_args: Dict


def configure(
    *, buddy: fannypack.utils.Buddy, task: Type[tasks.Task], dataset_args: Dict,
):
    """Configure global settings for eval helpers.
    """
    assert issubclass(task, tasks.Task)
    assert isinstance(buddy.model, diffbayes.base.Filter)
    globals()["buddy"] = buddy
    globals()["filter_model"] = cast(diffbayes.base.Filter, buddy.model)
    globals()["task"] = task
    globals()["dataset_args"] = dataset_args


def log_eval(measurement_initialize=False) -> None:
    """Evaluate a filter, print out + log metrics to Tensorboard.
    """
    results = run_eval(measurement_initialize)
    with buddy.log_scope("eval"):
        for key, value in results.items():
            if type(value) == float:
                buddy.log_scalar(key, value)


def run_eval_stats(*eval_args, **eval_kwargs) -> Dict[str, float]:
    all_results: Dict[str, List[float]] = {}

    # Evaluating...
    for i in range(20):
        results = run_eval(*eval_args, **eval_kwargs)

        # We don't care about raw RMSE
        results.pop("raw_rmse")

        # Everything else should be a float
        for v in results.values():
            assert isinstance(v, float)

        fannypack.utils.SliceWrapper(all_results).append(results)

    # Compute means, standard deviations
    results_stats = {}
    for k, v in all_results.items():
        results_stats[f"{k}_mean"] = float(np.array(v).mean())
        results_stats[f"{k}_std"] = float(np.array(v).std())

    for k, v in results_stats.items():
        print(f"{k}: {v}")

    # Return everything as a float
    return results_stats


def run_eval(measurement_initialize=False, eval_dynamics=False) -> Dict[str, float]:
    """Evaluate a filter, print out + return metrics.
    """
    assert isinstance(filter_model, diffbayes.base.Filter)

    # Get eval trajectories
    trajectories = globals()["task"].get_eval_trajectories(**dataset_args)
    assert type(trajectories) == list

    # Put model in eval mode
    filter_model.eval()

    with torch.no_grad():

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

        if measurement_initialize and hasattr(
            filter_model, "measurement_initialize_beliefs"
        ):
            print("initialize with measurement")
            filter_model.measurement_initialize_beliefs(
                observations=fannypack.utils.SliceWrapper(observations)[0]
            )
        else:
            print("init with random")
            cov = (torch.eye(state_dim, device=device) * 0.1)[None, :, :].expand(
                (N, state_dim, state_dim)
            )
            filter_model.initialize_beliefs(
                mean=states[0], covariance=cov,
            )

        # Run filter
        if eval_dynamics:
            predicted_states, _scale_trils = filter_model.dynamics_model.forward_loop(
                initial_states=states[0], controls=controls[1:]
            )
        else:
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
        per_batch_mse = np.mean(
            fannypack.utils.to_numpy(
                predicted_states[start_truncation:] - true_states[start_truncation:]
            )
            ** 2,
            axis=0,
        )

        assert per_batch_mse.shape == (N, state_dim)
        raw_rmse = np.sqrt(np.mean(per_batch_mse, axis=0))

        ## Compute across trajectories
        # raw_rmse = np.mean(np.sqrt(per_batch_mse), axis=0)
        # raw_rmse_std = np.std(np.sqrt(per_batch_mse), axis=0)

    if task == tasks.DoorTask:
        rmse = raw_rmse * np.array([0.39479038, 0.05650279, 0.0565098])
        # rmse_std = raw_rmse_std * np.array([0.39479038, 0.05650279, 0.0565098])
        results = {
            "raw_rmse": [float(x) for x in raw_rmse],
            "theta_rmse_deg": float(rmse[0] * 180.0 / np.pi),
            # "theta_rmse_deg_std": float(rmse_std[0] * 180.0 / np.pi),
            "x_rmse_cm": float(rmse[1] * 100.0),
            # "x_rmse_cm_std": float(rmse_std[1] * 100.0),
            "y_rmse_cm": float(rmse[2] * 100.0),
            # "y_rmse_cm_std": float(rmse_std[2] * 100.0),
        }
        print()
        print("-----")
        print(f"Raw RMSE:   {results['raw_rmse']}")
        print("-----")
        print(f"Theta RMSE: {results['theta_rmse_deg']:.8f} degrees")
        # print(f"Theta RMSE std: {results['theta_rmse_deg_std']:.8f} degrees")
        print()
        print(f"X RMSE:     {results['x_rmse_cm']:.8f} cm")
        # print(f"X RMSE std:     {results['x_rmse_cm_std']:.8f} cm")
        print()
        print(f"Y RMSE:     {results['y_rmse_cm']:.8f} cm")
        # print(f"Y RMSE std:     {results['y_rmse_cm_std']:.8f} cm")
        print("-----")

    elif task == tasks.PushTask:
        # TODO: this will be slightly off for the kloss dataset.
        # Currently be corrected in post-processing.
        rmse = raw_rmse * np.array([0.0572766, 0.06118315])
        # rmse_std = raw_rmse_std * np.array([0.0572766, 0.06118315])
        results = {
            "raw_rmse": [float(x) for x in raw_rmse],
            "x_rmse_cm": float(rmse[0] * 100.0),
            # "x_rmse_cm_std": float(rmse_std[0] * 100.0),
            "y_rmse_cm": float(rmse[1] * 100.0),
            # "y_rmse_cm_std": float(rmse_std[1] * 100.0),
        }
        print()
        print("-----")
        print(f"Raw RMSE:   {results['raw_rmse']}")
        print("-----")
        print(f"X RMSE:     {results['x_rmse_cm']:.8f} cm")
        # print(f"X RMSE std:     {results['x_rmse_cm_std']:.8f} cm")
        print()
        print(f"Y RMSE:     {results['y_rmse_cm']:.8f} cm")
        # print(f"Y RMSE std:     {results['x_rmse_cm_std']:.8f} cm")
        print("-----")
    else:
        assert False, "Invalid task!"

    return results
