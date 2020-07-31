import argparse
import dataclasses
import datetime
from typing import cast

import crossmodal
import diffbayes
import fannypack
from crossmodal.base_models import CrossmodalKalmanFilter

Task = crossmodal.tasks.PushTask

# Move cache in case we're running on NFS (eg Juno), open PDB on quit
fannypack.data.set_cache_path(crossmodal.__path__[0] + "/../.cache")
fannypack.utils.pdb_safety_net()

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--experiment-name", type=str)
parser.add_argument("--checkpoint-label", type=str, default=None)
parser.add_argument("--save", action="store_true")
parser.add_argument("--original-experiment", type=str, default=None)
parser.add_argument("--weighting_type", type=str, default="softmax", choices=["softmax", "absolute"])
parser.add_argument("--curriculum", type=int, default=0)

Task.add_dataset_arguments(parser)
# Parse args
args = parser.parse_args()
# Create Buddy and read experiment metadata
buddy = fannypack.utils.Buddy(args.experiment_name)


if args.original_experiment is not None:
    buddy_original = fannypack.utils.Buddy(args.original_experiment)

    try:
        model_type = buddy_original.metadata["model_type"]
        dataset_args = buddy_original.metadata["dataset_args"]

        buddy.set_metadata(buddy_original.metadata)

    except:
        print("missing metadata")

else:
    try:
        model_type = buddy.metadata["model_type"]
        dataset_args = buddy.metadata["dataset_args"]
    except:
        print("missing metadata")

filter_model: diffbayes.base.Filter = Task.model_types[model_type]()
buddy.attach_model(filter_model)

# buddy.load_checkpoint(experiment_name=args.original_experiment, label=args.checkpoint_label)

buddy.load_checkpoint_module(
    source="filter_models",
    target="filter_models",
    label=args.checkpoint_label,
    experiment_name=args.original_experiment
)


buddy.add_metadata({"curriculum": args.curriculum})
buddy.add_metadata({"original_experiment": args.original_experiment})


# Load trajectories into memory
train_trajectories = Task.get_train_trajectories(**dataset_args)
eval_trajectories = Task.get_eval_trajectories(**dataset_args)

# Configure helpers
train_helpers = crossmodal.train_helpers
train_helpers.configure(buddy=buddy, trajectories=train_trajectories)

eval_helpers = crossmodal.eval_helpers
eval_helpers.configure(buddy=buddy, trajectories=eval_trajectories, task=Task)

# Run model-specific training curriculum
if isinstance(filter_model, crossmodal.push_models.PushCrossmodalKalmanFilter):
    filter_model.crossmodal_weight_model.weighting_type = args.weighting_type

    image_model = filter_model.filter_models[0]
    force_model = filter_model.filter_models[1]
    filter_model.enabled_models = [True, True]

    if args.curriculum == 0:

        fannypack.utils.unfreeze_module(filter_model.crossmodal_weight_model)
        fannypack.utils.freeze_module(filter_model.filter_models)

        train_helpers.train_cm_e2e(subsequence_length=4, epochs=5, batch_size=32,
                                optimizer_name="freeze_ekf")
        eval_helpers.log_eval()

        train_helpers.train_cm_e2e(subsequence_length=8, epochs=5, batch_size=32,
                                optimizer_name="freeze_ekf")
        eval_helpers.log_eval()

        train_helpers.train_cm_e2e(subsequence_length=16, epochs=5, batch_size=32,
                                optimizer_name="freeze_ekf")

        # Train with weights frozen
        fannypack.utils.freeze_module(filter_model.crossmodal_weight_model)
        fannypack.utils.unfreeze_module(filter_model.filter_models)

        train_helpers.train_cm_e2e(subsequence_length=3, epochs=5,
                                   batch_size=32, measurement_initialize=False,
                                   optimizer_name="freeze_weights")
        eval_helpers.log_eval()

        buddy.save_checkpoint("phase4-length3")

        for _ in range(3):
            train_helpers.train_cm_e2e(subsequence_length=4, epochs=5, batch_size=32,
                                       measurement_initialize=False, optimizer_name="freeze_weights")
            eval_helpers.log_eval()

        buddy.save_checkpoint("phase4-length4")

        for _ in range(2):
            train_helpers.train_cm_e2e(subsequence_length=6, epochs=5, batch_size=32,
                                       measurement_initialize=False, optimizer_name="freeze_weights")
            eval_helpers.log_eval()

        buddy.save_checkpoint("phase4-length6")

        # Train everything end to end
        fannypack.utils.unfreeze_module(filter_model)
        for _ in range(2):
            train_helpers.train_cm_e2e(
                subsequence_length=6, epochs=5, batch_size=32, measurement_initialize=False
            )
            eval_helpers.log_eval()
        buddy.save_checkpoint("phase5-length6")

        # train using one loss
        for _ in range(2):
            train_helpers.train_e2e(
                subsequence_length=6, epochs=5, batch_size=32, measurement_initialize=False
            )
            eval_helpers.log_eval()

        eval_helpers.log_eval()
        print("kalman e2e")
        buddy.save_checkpoint("phase5-done")

    elif args.curriculum == 1:

        fannypack.utils.unfreeze_module(filter_model.crossmodal_weight_model)
        fannypack.utils.freeze_module(filter_model.filter_models)

        train_helpers.train_cm_e2e(subsequence_length=4, epochs=2, batch_size=32,
                                   optimizer_name="freeze_ekf")
        eval_helpers.log_eval()

        train_helpers.train_cm_e2e(subsequence_length=8, epochs=2, batch_size=32,
                                   optimizer_name="freeze_ekf")
        eval_helpers.log_eval()

        train_helpers.train_cm_e2e(subsequence_length=16, epochs=2, batch_size=32,
                                   optimizer_name="freeze_ekf")

        # Train with weights frozen
        fannypack.utils.freeze_module(filter_model.crossmodal_weight_model)
        fannypack.utils.unfreeze_module(filter_model.filter_models)

        train_helpers.train_cm_e2e(subsequence_length=3, epochs=2,
                                   batch_size=32, measurement_initialize=False,
                                   optimizer_name="freeze_weights")
        eval_helpers.log_eval()

        buddy.save_checkpoint("phase4-length3")

        for _ in range(3):
            train_helpers.train_cm_e2e(subsequence_length=4, epochs=2, batch_size=32,
                                       measurement_initialize=False, optimizer_name="freeze_weights")
            eval_helpers.log_eval()

        buddy.save_checkpoint("phase4-length4")

        for _ in range(2):
            train_helpers.train_cm_e2e(subsequence_length=6, epochs=2, batch_size=32,
                                       measurement_initialize=False, optimizer_name="freeze_weights")
            eval_helpers.log_eval()

        buddy.save_checkpoint("phase4-length6")

        # retrain weights again

        fannypack.utils.unfreeze_module(filter_model.crossmodal_weight_model)
        fannypack.utils.freeze_module(filter_model.filter_models)
        train_helpers.train_cm_e2e(subsequence_length=4, epochs=2, batch_size=32,
                                   optimizer_name="freeze_ekf")
        eval_helpers.log_eval()

        train_helpers.train_cm_e2e(subsequence_length=8, epochs=2, batch_size=32,
                                   optimizer_name="freeze_ekf")
        eval_helpers.log_eval()

        train_helpers.train_cm_e2e(subsequence_length=16, epochs=2, batch_size=32,
                                   optimizer_name="freeze_ekf")

        buddy.save_checkpoint("phase4.5-freezeweights2")

        # Train with weights frozen
        fannypack.utils.freeze_module(filter_model.crossmodal_weight_model)
        fannypack.utils.unfreeze_module(filter_model.filter_models)

        train_helpers.train_cm_e2e(subsequence_length=3, epochs=5,
                                   batch_size=32, measurement_initialize=False,
                                   optimizer_name="freeze_weights")
        eval_helpers.log_eval()

        buddy.save_checkpoint("phase5-length3")

        for _ in range(3):
            train_helpers.train_cm_e2e(subsequence_length=4, epochs=5, batch_size=32,
                                       measurement_initialize=False, optimizer_name="freeze_weights")
            eval_helpers.log_eval()

        buddy.save_checkpoint("phase5-length4")

        for _ in range(2):
            train_helpers.train_cm_e2e(subsequence_length=6, epochs=5, batch_size=32,
                                       measurement_initialize=False, optimizer_name="freeze_weights")
            eval_helpers.log_eval()

        buddy.save_checkpoint("phase5-length6")


        # Train everything end to end
        fannypack.utils.unfreeze_module(filter_model)
        for _ in range(2):
            train_helpers.train_cm_e2e(
                subsequence_length=6, epochs=5, batch_size=32, measurement_initialize=False
            )
            eval_helpers.log_eval()
        buddy.save_checkpoint("phase6-length6")

        # train using one loss
        for _ in range(2):
            train_helpers.train_e2e(
                subsequence_length=6, epochs=5, batch_size=32, measurement_initialize=False
            )
            eval_helpers.log_eval()

        eval_helpers.log_eval()
        print("kalman e2e")
        buddy.save_checkpoint("phase6-done")

    elif args.curriculum == 2:
        fannypack.utils.unfreeze_module(filter_model)
        train_helpers.train_cm_e2e(subsequence_length=3, epochs=5,
                                   batch_size=32, measurement_initialize=False,
                                   )
        eval_helpers.log_eval()

        buddy.save_checkpoint("phase4-length3")

        for _ in range(4):
            train_helpers.train_cm_e2e(subsequence_length=4, epochs=5, batch_size=32,
                                       measurement_initialize=False, )
            eval_helpers.log_eval()

        buddy.save_checkpoint("phase4-length4")

        for _ in range(4):
            train_helpers.train_cm_e2e(subsequence_length=6, epochs=5, batch_size=32,
                                       measurement_initialize=False, )
            eval_helpers.log_eval()

        buddy.save_checkpoint("phase4-length6")

        for _ in range(4):
            train_helpers.train_e2e(
                subsequence_length=6, epochs=5, batch_size=32, measurement_initialize=False
            )
            eval_helpers.log_eval()

        buddy.save_checkpoint("phase5-oneloss")



else:
    assert False, "No training curriculum found for model type"

# Add training end time
# buddy.add_metadata(
#     {"train_end_time": datetime.datetime.now().strftime("%b %d, %Y @ %-H:%M:%S"),}
# )

# Eval model when done
eval_results = crossmodal.eval_helpers.run_eval()
buddy.add_metadata({"eval_results": eval_results})

# python scripts/push_task/train_weights.py --checkpoint-label finetune_phase4-freeze --original-experiment 0729_pushreal_weights_cmekf_sm_2_trainlonger --experiment-name 0729_pushreal_weights_cmekf_sm_9_3losses

# python scripts/push_task/train_weights.py --checkpoint-label finetune_phase4-freeze --original-experiment  0729_pushreal_weights_cmekf_sm_1_trainunimodal  --experiment-name 0729_pushreal_weights_cmekf_sm_6_freezedyn

# python scripts/push_task/train_weights.py --checkpoint-label phase3-force --original-experiment 0728_pushreal_cmekf_0 --experiment-name 0729_pushreal_weights_cmekf_sm_2_trainlonger