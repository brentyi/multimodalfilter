import argparse
import dataclasses

import crossmodal
import diffbayes
import fannypack

# Move cache in case we're running on NFS (eg Juno), open PDB on quit
fannypack.data.set_cache_path(crossmodal.__path__[0] + "/../.cache")
fannypack.utils.pdb_safety_net()

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--experiment-name", type=str)
parser.add_argument("--checkpoint-label", type=str, default=None)
parser.add_argument("--save", action="store_true")
args = parser.parse_args()

# Create and validate Buddy
buddy = fannypack.utils.Buddy(args.experiment_name)
assert "model_type" in buddy.metadata
assert "dataset_args" in buddy.metadata

# Load model using experiment metadata
filter_model: diffbayes.base.Filter = crossmodal.door_models.model_types[
    buddy.metadata["model_type"]
]()
buddy.attach_model(filter_model)
buddy.load_checkpoint(label=args.checkpoint_label)

# Load trajectories using experiment metadata
trajectories = crossmodal.door_data.load_trajectories(
    "panda_door_pull_10.hdf5",
    "panda_door_push_10.hdf5",
    **buddy.metadata["dataset_args"]
)

# Run eval
filter_model.eval()
eval_results = crossmodal.eval_helpers.eval_filter(filter_model, trajectories)

# Save eval results
if args.save:
    buddy.add_metadata({"eval_results": dataclasses.asdict(eval_results)})
