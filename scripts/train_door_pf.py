import argparse

import crossmodal
import fannypack

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str, required=True)
args = parser.parse_args()

# Move cache in case we're running on NFS (eg Juno), open PDB on quit
fannypack.data.set_cache_path(crossmodal.__path__[0] + "/../.cache")
fannypack.utils.pdb_safety_net()

# Load training data
trajectories = crossmodal.door_data.load_trajectories(
    "panda_door_pull_300.hdf5", "panda_door_push_300.hdf5"
)

# Create model, Buddy
filter_model = crossmodal.door_particle_filter.DoorParticleFilter()
buddy = fannypack.utils.Buddy(args.experiment_name, filter_model)
buddy.set_metadata({"model_type": "particle_filter", "dataset_args": {}})

# Configure training helpers
train = crossmodal.door_train
train.configure(buddy=buddy, filter_model=filter_model, trajectories=trajectories)

# Run training curriculum
train.train_dynamics_single_step(epochs=5)
buddy.save_checkpoint("phase0")

train.train_dynamics_recurrent(subsequence_length=10, epochs=5)
train.train_dynamics_recurrent(subsequence_length=20, epochs=5)
buddy.save_checkpoint("phase1")

train.train_pf_measurement(epochs=3, batch_size=64)
buddy.save_checkpoint("phase2")

# Freeze dynamics
fannypack.utils.freeze_module(filter_model.dynamics_model)

filter_model.resample = False
train.train_e2e(subsequence_length=3, epochs=5)
train.train_e2e(subsequence_length=10, epochs=5)
train.train_e2e(subsequence_length=20, epochs=5)
buddy.save_checkpoint("phase3")

fannypack.utils.unfreeze_module(filter_model.dynamics_model)

# Load eval trajectories using experiment metadata
eval_trajectories = crossmodal.door_data.load_trajectories(
    "panda_door_pull_10.hdf5",
    "panda_door_push_10.hdf5",
    **(buddy.metadata["dataset_args"])
)

# Run eval
filter_model.resample = True
crossmodal.door_eval.eval_model(filter_model, eval_trajectories)
