import argparse
import dataclasses
import datetime
from typing import cast

import crossmodal
import fannypack
from crossmodal.base_models import CrossmodalParticleFilterMeasurementModel

# Index of trainable models
model_types = crossmodal.door_models.model_types

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--model-type", type=str, required=True, choices=model_types.keys())
parser.add_argument("--experiment-name", type=str, required=True)
parser.add_argument("--notes", type=str, default="(none)")
crossmodal.door_data.add_dataset_arguments(parser)

# Parse args
args = parser.parse_args()
model_type = args.model_type
dataset_args = crossmodal.door_data.get_dataset_args(args)

# Move cache in case we're running on NFS (eg Juno), open PDB on quit
fannypack.data.set_cache_path(crossmodal.__path__[0] + "/../.cache")
fannypack.utils.pdb_safety_net()

# Load trajectories into memory
train_trajectories = crossmodal.door_data.load_trajectories(
    "panda_door_pull_100.hdf5", "panda_door_push_100.hdf5", **dataset_args
)
eval_trajectories = crossmodal.door_data.load_trajectories(
    "panda_door_pull_10.hdf5", "panda_door_push_10.hdf5", **dataset_args
)

# Create model, Buddy
filter_model = model_types[model_type]()
buddy = fannypack.utils.Buddy(args.experiment_name, filter_model)
buddy.set_metadata(
    {
        "model_type": model_type,
        "dataset_args": dataset_args,
        "train_start_time": datetime.datetime.now().strftime("%b %d, %Y @ %-H:%M:%S"),
        "commit_hash": fannypack.utils.get_git_commit_hash(crossmodal.__file__),
        "notes": args.notes,
    }
)

# Configure helpers
train_helpers = crossmodal.train_helpers
train_helpers.configure(buddy=buddy, trajectories=train_trajectories)

eval_helpers = crossmodal.eval_helpers
eval_helpers.configure(buddy=buddy, trajectories=eval_trajectories)

# Run model-specific training curriculum
if isinstance(filter_model, crossmodal.door_models.DoorLSTMFilter):
    # Pre-train for single-step prediction
    train_helpers.train_e2e(subsequence_length=2, epochs=2, batch_size=32)
    buddy.save_checkpoint("phase0")

    # Train on longer sequences
    train_helpers.train_e2e(subsequence_length=3, epochs=5, batch_size=32)
    eval_helpers.log_eval()
    train_helpers.train_e2e(subsequence_length=8, epochs=5, batch_size=32)
    eval_helpers.log_eval()
    train_helpers.train_e2e(subsequence_length=16, epochs=20, batch_size=32)
    eval_helpers.log_eval()
    buddy.save_checkpoint("phase1")

elif isinstance(filter_model, crossmodal.door_models.DoorParticleFilter):
    # Pre-train dynamics (single-step)
    train_helpers.train_pf_dynamics_single_step(epochs=5)
    buddy.save_checkpoint("phase0")

    # Pre-train dynamics (recurrent)
    train_helpers.train_pf_dynamics_recurrent(subsequence_length=8, epochs=5)
    train_helpers.train_pf_dynamics_recurrent(subsequence_length=16, epochs=10)
    eval_helpers.log_eval()
    buddy.save_checkpoint("phase1")

    # Pre-train measurement model
    train_helpers.train_pf_measurement(epochs=5, batch_size=64)
    train_helpers.train_e2e(subsequence_length=16, epochs=20, batch_size=32)
    eval_helpers.log_eval()
    buddy.save_checkpoint("phase2")

    # Train E2E (w/ frozen dynamics)
    fannypack.utils.freeze_module(filter_model.dynamics_model)
    train_helpers.train_e2e(subsequence_length=3, epochs=5, batch_size=32)
    eval_helpers.log_eval()
    train_helpers.train_e2e(subsequence_length=8, epochs=5, batch_size=32)
    eval_helpers.log_eval()
    for _ in range(4):
        train_helpers.train_e2e(subsequence_length=16, epochs=5, batch_size=32)
        eval_helpers.log_eval()
    buddy.save_checkpoint("phase3")

elif isinstance(filter_model, crossmodal.door_models.DoorCrossmodalParticleFilter):
    # Pull out measurement model, freeze crossmodal weights
    measurement_model: CrossmodalParticleFilterMeasurementModel = cast(
        CrossmodalParticleFilterMeasurementModel, filter_model.measurement_model,
    )
    fannypack.utils.freeze_module(measurement_model.crossmodal_weight_model)

    # Pre-train dynamics (single-step)
    train_helpers.train_pf_dynamics_single_step(epochs=5)
    buddy.save_checkpoint("phase0")

    # Pre-train dynamics (recurrent)
    train_helpers.train_pf_dynamics_recurrent(subsequence_length=8, epochs=5)
    train_helpers.train_pf_dynamics_recurrent(subsequence_length=16, epochs=5)
    buddy.save_checkpoint("phase1")

    # Freeze dynamics
    fannypack.utils.freeze_module(filter_model.dynamics_model)

    # Pre-train measurement model (image)
    measurement_model._enabled_models = [True, False]
    train_helpers.train_pf_measurement(epochs=3, batch_size=64)
    train_helpers.train_e2e(subsequence_length=3, epochs=5, batch_size=32)
    train_helpers.train_e2e(subsequence_length=8, epochs=5, batch_size=32)
    train_helpers.train_e2e(subsequence_length=16, epochs=20, batch_size=32)
    buddy.save_checkpoint("phase2")

    # Pre-train measurement model (proprioception + haptics)
    measurement_model._enabled_models = [False, True]
    train_helpers.train_pf_measurement(epochs=3, batch_size=64)
    train_helpers.train_e2e(subsequence_length=3, epochs=5, batch_size=32)
    eval_helpers.log_eval()
    train_helpers.train_e2e(subsequence_length=8, epochs=5, batch_size=32)
    eval_helpers.log_eval()
    train_helpers.train_e2e(subsequence_length=16, epochs=20, batch_size=32)
    eval_helpers.log_eval()
    buddy.save_checkpoint("phase3")

    # Enable both measurement models
    measurement_model._enabled_models = [True, True]

    # Unfreeze weight model, freeze measurement model
    fannypack.utils.unfreeze_module(measurement_model.crossmodal_weight_model)
    fannypack.utils.freeze_module(measurement_model.measurement_models)

    # Train everything end-to-end
    train_helpers.train_e2e(subsequence_length=3, epochs=5, batch_size=32)
    eval_helpers.log_eval()
    train_helpers.train_e2e(subsequence_length=8, epochs=5, batch_size=32)
    eval_helpers.log_eval()
    for _ in range(4):
        train_helpers.train_e2e(subsequence_length=16, epochs=5, batch_size=32)
        eval_helpers.log_eval()
    buddy.save_checkpoint("phase4")

else:
    assert False, "No training curriculum found for model type"

# Add training end time
buddy.add_metadata(
    {"train_end_time": datetime.datetime.now().strftime("%b %d, %Y @ %-H:%M:%S"),}
)

# Eval model when done
eval_results = crossmodal.eval_helpers.run_eval()
buddy.add_metadata({"eval_results": dataclasses.asdict(eval_results)})
