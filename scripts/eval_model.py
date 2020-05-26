import argparse

import numpy as np
import torch
from tqdm.auto import tqdm

import crossmodal
import diffbayes
import fannypack as fp

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str)
args = parser.parse_args()

# Move cache in case we're running on NFS (eg Juno), open PDB on quit
fp.data.set_cache_path(crossmodal.__path__[0] + "/../.cache")
fp.utils.pdb_safety_net()

# Create Buddy and use metadata to load model and dataset
buddy = fp.utils.Buddy(args.experiment_name)
assert "model_type" in buddy.metadata
assert "dataset_args" in buddy.metadata
filter_model: diffbayes.base.Filter = {
    "lstm": crossmodal.door_lstm.DoorLSTMFilter,
    "particle_filter": crossmodal.door_particle_filter.DoorParticleFilter,
}[buddy.metadata["model_type"]]()
buddy.attach_model(filter_model)
buddy.load_checkpoint()

trajectories = crossmodal.door_data.load_trajectories(
    "panda_door_pull_10.hdf5",
    "panda_door_push_10.hdf5",
    **buddy.metadata["dataset_args"]
)

# Convert list of trajectories -> batch
#
# Note that we need to jump through some hoops because the observations are stored as
# dictionaries
states = []
observations = fp.utils.SliceWrapper({})
controls = []
min_timesteps = min([s.shape[0] for s, o, c in trajectories])
for (s, o, c) in trajectories:
    # Update batch
    states.append(s[:min_timesteps])
    observations.append(fp.utils.SliceWrapper(o)[:min_timesteps])
    controls.append(c[:min_timesteps])

# Numpy -> torch, create batch axis @ index 1
stack_fn = lambda list_value: np.stack(list_value, axis=1)
states, observations, controls = fp.utils.to_torch(
    [stack_fn(states), observations.map(stack_fn), stack_fn(controls)],
    device=buddy.device,
)

# Get sequence length (T) and batch size (N) dimensions
assert states.shape[:2] == controls.shape[:2]
assert states.shape[:2] == fp.utils.SliceWrapper(observations).shape[:2]
T, N = states.shape[:2]

# Initialize beliefs
state_dim = filter_model.state_dim
cov = (torch.eye(state_dim, device=buddy.device) * 0.1)[None, :, :].expand(
    (N, state_dim, state_dim)
)
filter_model.initialize_beliefs(
    mean=states[0], covariance=cov,
)

# Run filter
with torch.no_grad():
    predicted_states = filter_model.forward_loop(
        observations=fp.utils.SliceWrapper(observations)[1:], controls=controls[1:],
    )

# Validate predicted states
T = predicted_states.shape[0]
assert predicted_states.shape == (T, N, state_dim)

# Compute & update errors
true_states = states[1:]
mse = np.mean(
    fp.utils.to_numpy(predicted_states - true_states).reshape((-1, state_dim)) ** 2,
    axis=0,
)
rmse = np.sqrt(mse / len(trajectories))
print()
print()
print(rmse)
