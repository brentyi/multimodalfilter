import argparse
import datetime
import dataclasses
from typing import cast

import crossmodal
import diffbayes
import fannypack
from crossmodal.base_models import CrossmodalKalmanFilter

# Move cache in case we're running on NFS (eg Juno), open PDB on quit
fannypack.data.set_cache_path(crossmodal.__path__[0] + "/../.cache")
fannypack.utils.pdb_safety_net()

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--experiment-name", type=str)
parser.add_argument("--checkpoint-label", type=str, default=None)
parser.add_argument("--save", action="store_true")
parser.add_argument("--original-experiment", type=str, default=None)
parser.add_argument("--unimodal", action="store_true")
crossmodal.push_data.add_dataset_arguments(parser)
# Parse args
args = parser.parse_args()
# Create Buddy and read experiment metadata
buddy = fannypack.utils.Buddy(args.experiment_name)

# --original-experiment door_cmekf_measinit --checkpoint-label phase3 --experiment-name door_cmekf_measinit_finetune

if args.original_experiment is not None:
    buddy_original = fannypack.utils.Buddy(args.original_experiment)

    try:
        model_type = buddy_original.metadata["model_type"]
        dataset_args = buddy_original.metadata["dataset_args"]

        buddy.set_metadata(buddy_original.metadata)

    except:
        dataset_args = crossmodal.push_data.get_dataset_args(args)
        model_type = "DoorCrossmodalKalmanFilter"

else:
    try:
        model_type = buddy.metadata["model_type"]
        dataset_args = buddy.metadata["dataset_args"]
    except:
        dataset_args = crossmodal.push_data.get_dataset_args(args)
        model_type = "DoorCrossmodalKalmanFilter"

if args.unimodal:
    # going from crossmodal to unimodal models, loading just the filter models

    filter_model: diffbayes.base.Filter = crossmodal.door_models.DoorUnimodalKalmanFilter()
    buddy.attach_model(filter_model)
    buddy.load_checkpoint_module(
        source="filter_models",
        target="filter_models",
        label=args.checkpoint_label,
        experiment_name=args.original_experiment
    )
else:
    # Load model using experiment metadata
    filter_model: diffbayes.base.Filter = crossmodal.door_models.model_types[model_type]()
    buddy.attach_model(filter_model)
    buddy.load_checkpoint(label=args.checkpoint_label, experiment_name=args.original_experiment)

# Load trajectories into memory
train_trajectories = crossmodal.door_data.load_trajectories(
    "panda_door_pull_100.hdf5", "panda_door_push_100.hdf5", **dataset_args
)
eval_trajectories = crossmodal.door_data.load_trajectories(
    "panda_door_pull_10.hdf5", "panda_door_push_10.hdf5", **dataset_args
)

# Configure helpers
train_helpers = crossmodal.train_helpers
train_helpers.configure(buddy=buddy, trajectories=train_trajectories)

eval_helpers = crossmodal.eval_helpers
eval_helpers.configure(buddy=buddy, trajectories=eval_trajectories, task="door")

# Run model-specific training curriculum
if isinstance(filter_model, crossmodal.door_models.DoorCrossmodalKalmanFilter):
    image_model = filter_model.filter_models[0]
    force_model = filter_model.filter_models[1]

    if args.checkpoint_label == "phase2":
        # Pre-train kalman filter (image)
        filter_model.enabled_models = [True, False]
        train_helpers.train_e2e(subsequence_length=4, epochs=3, batch_size=32,
                                optimizer_name="image_ekf")
        eval_helpers.log_eval()
        train_helpers.train_e2e(subsequence_length=8, epochs=3, batch_size=32,
                                optimizer_name="image_ekf")
        eval_helpers.log_eval()
        train_helpers.train_e2e(subsequence_length=16, epochs=5, batch_size=32,
                                optimizer_name="image_ekf")
        eval_helpers.log_eval()
        buddy.save_checkpoint("phase3-image")

        # Pre-train kalman filter (proprioception + haptics)
        filter_model.enabled_models = [False, True]
        train_helpers.train_e2e(subsequence_length=4, epochs=3, batch_size=32,
                                optimizer_name="force_ekf")
        eval_helpers.log_eval()
        train_helpers.train_e2e(subsequence_length=8, epochs=3, batch_size=32,
                                optimizer_name="force_ekf")
        eval_helpers.log_eval()
        train_helpers.train_e2e(subsequence_length=16, epochs=5, batch_size=32,
                                optimizer_name="force_ekf")
        eval_helpers.log_eval()
        buddy.save_checkpoint("phase3-force")
    # Enable both filter models
    filter_model.enabled_models = [True, True]

    # Unfreeze weight model, freeze filter model
    fannypack.utils.unfreeze_module(filter_model.crossmodal_weight_model)
    fannypack.utils.freeze_module(filter_model.filter_models)
    train_helpers.train_e2e(subsequence_length=3, epochs=1, batch_size=32,
                            optimizer_name="freeze_ekf")
    buddy.save_checkpoint("finetune_phase4-freeze")

    # Train everything end-to-end
    fannypack.utils.unfreeze_module(filter_model.filter_models)

    train_helpers.train_e2e(subsequence_length=3, epochs=5,
                            batch_size=32, measurement_initialize=False)
    eval_helpers.log_eval()
    buddy.save_checkpoint("finetune_phase4-length3")

    buddy.set_regularization_weight(optimizer_name="train_filter_recurrent", value=0.0001)

    for _ in range(3):
        train_helpers.train_e2e(subsequence_length=4, epochs=5, batch_size=32,
                                measurement_initialize=False)
        eval_helpers.log_eval()
    buddy.save_checkpoint("finetune_phase4-length4")

    for _ in range(2):
        train_helpers.train_e2e(subsequence_length=6, epochs=5, batch_size=32,
                                measurement_initialize=False)
        eval_helpers.log_eval()
        print("kalman e2e")

    buddy.save_checkpoint("finetune_phase4-done")


elif isinstance(filter_model, crossmodal.door_models.DoorUnimodalKalmanFilter):

    if args.checkpoint_label == "phase2":
        # Pre-train kalman filter (image)
        filter_model.enabled_models = [True, False]
        train_helpers.train_e2e(subsequence_length=4, epochs=3, batch_size=32,
                                optimizer_name="image_ekf")
        eval_helpers.log_eval()
        train_helpers.train_e2e(subsequence_length=8, epochs=3, batch_size=32,
                                optimizer_name="image_ekf")
        eval_helpers.log_eval()
        train_helpers.train_e2e(subsequence_length=16, epochs=5, batch_size=32,
                                optimizer_name="image_ekf")
        eval_helpers.log_eval()
        buddy.save_checkpoint("phase3-image")

        # Pre-train kalman filter (proprioception + haptics)
        filter_model.enabled_models = [False, True]
        train_helpers.train_e2e(subsequence_length=4, epochs=3, batch_size=32,
                                optimizer_name="force_ekf")
        eval_helpers.log_eval()
        train_helpers.train_e2e(subsequence_length=8, epochs=3, batch_size=32,
                                optimizer_name="force_ekf")
        eval_helpers.log_eval()
        train_helpers.train_e2e(subsequence_length=16, epochs=5, batch_size=32,
                                optimizer_name="force_ekf")
        eval_helpers.log_eval()
        buddy.save_checkpoint("phase3-force")

    filter_model.enabled_models = [True, True]

    # Train everything end-to-end
    fannypack.utils.unfreeze_module(filter_model.filter_models)

    train_helpers.train_e2e(subsequence_length=3, epochs=5,
                            batch_size=32, measurement_initialize=False)
    eval_helpers.log_eval()
    buddy.save_checkpoint("finetune_phase4-length3")

    buddy.set_regularization_weight(optimizer_name="train_filter_recurrent", value=0.0001)

    for _ in range(3):
        train_helpers.train_e2e(subsequence_length=4, epochs=5, batch_size=32,
                                measurement_initialize=False)
        eval_helpers.log_eval()
    buddy.save_checkpoint("finetune_phase4-length4")

    for _ in range(2):
        train_helpers.train_e2e(subsequence_length=6, epochs=5, batch_size=32,
                                measurement_initialize=False)
        eval_helpers.log_eval()
        print("kalman e2e")

    buddy.save_checkpoint("finetune_phase4-done")

elif isinstance(filter_model, crossmodal.door_models.DoorKalmanFilter):
    fannypack.utils.freeze_module(filter_model.dynamics_model)

    # Train E2E
    train_helpers.train_e2e(subsequence_length=4, epochs=5, batch_size=32,
                                measurement_initialize=True)
    eval_helpers.log_eval()
    train_helpers.train_e2e(subsequence_length=8, epochs=5, batch_size=32,
                                measurement_initialize=True)
    eval_helpers.log_eval()
    for _ in range(4):
        train_helpers.train_e2e(subsequence_length=16, epochs=5, batch_size=32,
                                measurement_initialize=True)
        eval_helpers.log_eval()
    buddy.save_checkpoint("finetune-done")

else:
    assert False, "No training curriculum found for model type"

# Add training end time
# buddy.add_metadata(
#     {"train_end_time": datetime.datetime.now().strftime("%b %d, %Y @ %-H:%M:%S"),}
# )

# Eval model when done
# Eval model when done
eval_results = crossmodal.eval_helpers.run_eval()
buddy.add_metadata({"eval_results": eval_results})
