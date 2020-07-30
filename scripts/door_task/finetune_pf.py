# Our normal CMPF training curriculum doesn't ever unfreeze *both* of our measurement
# models and train end-to-end
#
# Sole purpose of this script is to do that :)

import argparse
import dataclasses
import datetime
from typing import cast

import fannypack

import crossmodal
import diffbayes

Task = crossmodal.tasks.DoorTask

# Move cache in case we're running on NFS (eg Juno), open PDB on quit
fannypack.data.set_cache_path(crossmodal.__path__[0] + "/../.cache")
fannypack.utils.pdb_safety_net()

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--experiment-name", type=str, required=True)
parser.add_argument("--variant", type=str, required=True)
parser.add_argument("--checkpoint-label", type=str, default=None)
args = parser.parse_args()

# Create Buddy and read experiment metadata
buddy = fannypack.utils.Buddy(args.experiment_name + "_" + args.variant)
buddy.load_metadata(experiment_name=args.experiment_name)
model_type = buddy.metadata["model_type"]
dataset_args = buddy.metadata["dataset_args"]

# Load model using experiment metadata
filter_model: diffbayes.base.Filter = Task.model_types[model_type]()
buddy.attach_model(filter_model)
try:
    buddy.load_checkpoint()
except FileNotFoundError:
    buddy.load_checkpoint(
        experiment_name=args.experiment_name, label=args.checkpoint_label
    )

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
if isinstance(filter_model, crossmodal.door_models.DoorParticleFilter):

    learning_rate = 1e-5

    def warmup_lr(optimizer_name="train_filter_recurrent", warmup_steps=100):
        start_steps = buddy.optimizer_steps

        buddy.set_learning_rate(
            optimizer_name="train_filter_recurrent",
            value=lambda steps: min(
                learning_rate, learning_rate * (steps - start_steps) / warmup_steps
            ),
        )

    # Pull out measurement model, freeze crossmodal weights
    measurement_model: diffbayes.base.ParticleFilterMeasurementModel = cast(
        diffbayes.base.ParticleFilterMeasurementModel, filter_model.measurement_model,
    )

    # Freeze dynamics
    fannypack.utils.freeze_module(filter_model.dynamics_model)

    # Make sure measurement model is unfrozen
    fannypack.utils.unfreeze_module(measurement_model)


    # Train with unfrozen measurement models
    eval_helpers.log_eval()
    warmup_lr()
    for _ in range(30):
        train_helpers.train_e2e(
            subsequence_length=16,
            epochs=1,
            batch_size=32,
            optimizer_name="train_filter_recurrent",
        )
        eval_helpers.log_eval()
    buddy.save_checkpoint()

else:
    assert False, "No training curriculum found for model type"

# Add training end time
buddy.add_metadata(
    {"finetune_end_time": datetime.datetime.now().strftime("%b %d, %Y @ %-H:%M:%S"),}
)

# Eval model when done
eval_results = crossmodal.eval_helpers.run_eval()
buddy.add_metadata({"eval_results": eval_results})
