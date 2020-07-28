import argparse
import dataclasses
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
parser.add_argument("--experiment-name", type=str)
parser.add_argument("--checkpoint-label", type=str, default=None)
parser.add_argument("--save", action="store_true")
args = parser.parse_args()

# Create Buddy and read experiment metadata
buddy = fannypack.utils.Buddy(args.experiment_name)
model_type = buddy.metadata["model_type"]
dataset_args = buddy.metadata["dataset_args"]

# Load model using experiment metadata
filter_model: diffbayes.base.Filter = Task.model_types[model_type]()
buddy.attach_model(filter_model)
buddy.load_checkpoint(label=args.checkpoint_label)

# Load trajectories into memory
train_trajectories = Task.get_train_trajectories(**dataset_args)
eval_trajectories = Task.get_eval_trajectories(**dataset_args)

# Configure helpers
train_helpers = crossmodal.train_helpers
train_helpers.configure(buddy=buddy, trajectories=train_trajectories)

eval_helpers = crossmodal.eval_helpers
eval_helpers.configure(buddy=buddy, trajectories=eval_trajectories, task=Task)

# Run model-specific training curriculum
if isinstance(filter_model, crossmodal.door_models.DoorCrossmodalParticleFilter):
    # Pull out measurement model, freeze crossmodal weights
    measurement_model: CrossmodalParticleFilterMeasurementModel = cast(
        CrossmodalParticleFilterMeasurementModel, filter_model.measurement_model,
    )
    fannypack.utils.freeze_module(measurement_model.crossmodal_weight_model)

    # Pre-train dynamics (single-step)
    train_helpers.train_pf_dynamics_single_step(epochs=5)
    eval_helpers.log_eval()
    buddy.save_checkpoint("finetune-phase0")

    # Pre-train dynamics (recurrent)
    train_helpers.train_pf_dynamics_recurrent(subsequence_length=4, epochs=5)
    eval_helpers.log_eval()
    train_helpers.train_pf_dynamics_recurrent(subsequence_length=8, epochs=5)
    eval_helpers.log_eval()
    train_helpers.train_pf_dynamics_recurrent(subsequence_length=16, epochs=5)
    eval_helpers.log_eval()
    buddy.save_checkpoint("finetune-phase1")

    # Freeze dynamics
    fannypack.utils.freeze_module(filter_model.dynamics_model)

    # Enable both measurement models
    measurement_model.enabled_models = [True, True]

    # Unfreeze weight model, freeze measurement model
    fannypack.utils.unfreeze_module(measurement_model.crossmodal_weight_model)
    fannypack.utils.unfreeze_module(measurement_model.measurement_models)

    # Train everything end-to-end
    train_helpers.train_e2e(subsequence_length=4, epochs=5, batch_size=32)
    eval_helpers.log_eval()
    train_helpers.train_e2e(subsequence_length=8, epochs=5, batch_size=32)
    eval_helpers.log_eval()
    train_helpers.train_e2e(subsequence_length=16, epochs=5, batch_size=32)
    eval_helpers.log_eval()
    buddy.save_checkpoint("finetune-phase2")

else:
    assert False, "No training curriculum found for model type"

# Add training end time
buddy.add_metadata(
    {"train_end_time": datetime.datetime.now().strftime("%b %d, %Y @ %-H:%M:%S"),}
)

# Eval model when done
eval_results = crossmodal.eval_helpers.run_eval()
buddy.add_metadata({"eval_results": dataclasses.asdict(eval_results)})
