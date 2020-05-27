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
filter_model = crossmodal.door_lstm.DoorLSTMFilter()
buddy = fannypack.utils.Buddy(args.experiment_name, filter_model)
buddy.set_metadata({"model_type": "lstm", "dataset_args": {}})

# Configure training helpers
train = crossmodal.door_train
train.configure(buddy=buddy, filter_model=filter_model, trajectories=trajectories)

# Run training curriculum
train.train_e2e(subsequence_length=2, batch_size=32, epochs=2)
buddy.save_checkpoint("phase0")

train.train_e2e(subsequence_length=16, batch_size=32, epochs=20)
buddy.save_checkpoint("phase1")

# Load eval trajectories using experiment metadata
eval_trajectories = crossmodal.door_data.load_trajectories(
    "panda_door_pull_10.hdf5",
    "panda_door_push_10.hdf5",
    **(buddy.metadata["dataset_args"])
)

# Run eval
filter_model.resample = True
crossmodal.door_eval.eval_model(filter_model, eval_trajectories)
