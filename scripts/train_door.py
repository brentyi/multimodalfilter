import argparse

import crossmodal
import fannypack

# Index of trainable models
model_types = crossmodal.door_models.model_types

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, required=True, choices=model_types.keys())
parser.add_argument("--experiment_name", type=str, required=True)
crossmodal.door_data.add_dataset_arguments(parser)
args = parser.parse_args()

# Get dataset args
dataset_args = crossmodal.door_data.get_dataset_args(args)

# Move cache in case we're running on NFS (eg Juno), open PDB on quit
fannypack.data.set_cache_path(crossmodal.__path__[0] + "/../.cache")
fannypack.utils.pdb_safety_net()

# Load training data
train_trajectories = crossmodal.door_data.load_trajectories(
    "panda_door_pull_300.hdf5", "panda_door_push_300.hdf5", **dataset_args
)

# Create model, Buddy
filter_model = model_types[args.model_type]()
buddy = fannypack.utils.Buddy(args.experiment_name, filter_model)
buddy.set_metadata({"model_type": args.model_type, "dataset_args": dataset_args})

# Configure training helpers
train_helpers = crossmodal.train_helpers
train_helpers.configure(
    buddy=buddy, filter_model=filter_model, trajectories=train_trajectories
)

# Run model-specific training curriculum
filter_model.train()
if args.model_type == "lstm":
    assert isinstance(filter_model, crossmodal.door_lstm.DoorLSTMFilter)

    # Pre-train for single-step prediction
    train_helpers.train_e2e(subsequence_length=2, epochs=2, batch_size=32)
    buddy.save_checkpoint("phase0")

    # Train on longer sequences
    train_helpers.train_e2e(subsequence_length=3, epochs=5, batch_size=32)
    train_helpers.train_e2e(subsequence_length=16, epochs=10, batch_size=32)
    buddy.save_checkpoint("phase1")

elif args.model_type == "particle_filter":
    assert isinstance(filter_model, crossmodal.door_particle_filter.DoorParticleFilter)

    # Pre-train dynamics (single-step)
    train_helpers.train_pf_dynamics_single_step(epochs=5)
    buddy.save_checkpoint("phase0")

    # Pre-train dynamics (recurrent)
    train_helpers.train_pf_dynamics_recurrent(subsequence_length=8, epochs=5)
    train_helpers.train_pf_dynamics_recurrent(subsequence_length=16, epochs=5)
    buddy.save_checkpoint("phase1")

    # Pre-train measurement model
    train_helpers.train_pf_measurement(epochs=3, batch_size=64)
    buddy.save_checkpoint("phase2")

    # Train E2E (w/ frozen dynamics)
    fannypack.utils.freeze_module(filter_model.dynamics_model)
    train_helpers.train_e2e(subsequence_length=3, epochs=5, batch_size=32)
    train_helpers.train_e2e(subsequence_length=8, epochs=5, batch_size=32)
    train_helpers.train_e2e(subsequence_length=16, epochs=5, batch_size=32)
    fannypack.utils.unfreeze_module(filter_model.dynamics_model)
    buddy.save_checkpoint("phase3")

# Eval model when done
eval_trajectories = crossmodal.door_data.load_trajectories(
    "panda_door_pull_10.hdf5", "panda_door_push_10.hdf5", **dataset_args
)
filter_model.eval()
crossmodal.eval_helpers.eval_filter(filter_model, eval_trajectories)
