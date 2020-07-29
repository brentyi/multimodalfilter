# Our normal CMPF training curriculum doesn't ever unfreeze *both* of our measurement
# models and train end-to-end
#
# Sole purpose of this script is to do that :)

import argparse
import dataclasses
import datetime
from typing import cast

import crossmodal
import diffbayes
import fannypack
from crossmodal.base_models import CrossmodalParticleFilterMeasurementModel

Task = crossmodal.tasks.DoorTask

# Move cache in case we're running on NFS (eg Juno), open PDB on quit
fannypack.data.set_cache_path(crossmodal.__path__[0] + "/../.cache")
fannypack.utils.pdb_safety_net()

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--experiment-name", type=str, required=True)
parser.add_argument("--checkpoint-label", type=str, default=None)
args = parser.parse_args()

# Create Buddy and read experiment metadata
buddy = fannypack.utils.Buddy(args.experiment_name + "_finetune")
buddy.load_metadata(experiment_name=args.experiment_name)
model_type = buddy.metadata["model_type"]
dataset_args = buddy.metadata["dataset_args"]

# Load model using experiment metadata
filter_model: diffbayes.base.Filter = Task.model_types[model_type]()
buddy.attach_model(filter_model)
# try:
#     buddy.load_checkpoint()
# except FileNotFoundError:
#     buddy.load_checkpoint(
#         experiment_name=args.experiment_name, label=args.checkpoint_label
#     )

# Load trajectories into memory
train_trajectories = Task.get_train_trajectories(**dataset_args)
eval_trajectories = Task.get_eval_trajectories(**dataset_args)

# Configure helpers
train_helpers = crossmodal.train_helpers
train_helpers.configure(buddy=buddy, trajectories=train_trajectories)

eval_helpers = crossmodal.eval_helpers
eval_helpers.configure(buddy=buddy, trajectories=eval_trajectories, task=Task)

# Add training start time
buddy.add_metadata(
    {"finetune_start_time": datetime.datetime.now().strftime("%b %d, %Y @ %-H:%M:%S"),}
)

# Run model-specific training curriculum
if isinstance(filter_model, crossmodal.door_models.DoorCrossmodalParticleFilter):
    # Pull out measurement model, freeze crossmodal weights
    measurement_model: CrossmodalParticleFilterMeasurementModel = cast(
        CrossmodalParticleFilterMeasurementModel, filter_model.measurement_model,
    )

    # Freeze dynamics
    fannypack.utils.freeze_module(filter_model.dynamics_model)

    # Enable both measurement models
    measurement_model.enabled_models = [True, True]

    # Make sure weight model, measurement model are unfrozen
    # (this should always be true)
    fannypack.utils.unfreeze_module(measurement_model.crossmodal_weight_model)
    fannypack.utils.unfreeze_module(measurement_model.measurement_models)

    def warmup_lr():
        start_steps = buddy.optimizer_steps
        warmup_steps = 200
        buddy.set_learning_rate(
            optimizer_name="train_filter_recurrent",
            value=lambda steps: min(1e-4, (steps - start_steps) / warmup_steps * 1e-4),
        )

    # Train with unfrozen measurement models
    eval_helpers.log_eval()
    warmup_lr()
    for _ in range(5):
        train_helpers.train_e2e(
            subsequence_length=8,
            epochs=1,
            batch_size=32,
            optimizer_name="train_filter_recurrent",
        )
        eval_helpers.log_eval()
    warmup_lr()
    for _ in range(10):
        train_helpers.train_e2e(
            subsequence_length=16,
            epochs=1,
            batch_size=32,
            optimizer_name="train_filter_recurrent",
        )
        eval_helpers.log_eval()
    buddy.save_checkpoint("finetune-phase2")

else:
    assert False, "No training curriculum found for model type"

# Add training end time
buddy.add_metadata(
    {"finetune_end_time": datetime.datetime.now().strftime("%b %d, %Y @ %-H:%M:%S"),}
)

# Eval model when done
eval_results = crossmodal.eval_helpers.run_eval()
buddy.add_metadata({"eval_results": eval_results})
