import argparse

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

# Create Buddy and read experiment metadata
buddy = fannypack.utils.Buddy(args.experiment_name)
model_type = buddy.metadata["model_type"]
dataset_args = buddy.metadata["dataset_args"]

# Load model using experiment metadata
filter_model: diffbayes.base.Filter = crossmodal.push_models.model_types[model_type]()
buddy.attach_model(filter_model)
buddy.load_checkpoint(label=args.checkpoint_label)

# Load trajectories using experiment metadata
eval_trajectories = crossmodal.push_data.load_trajectories(
    "gentle_push_10.hdf5", **dataset_args
)

# Run eval
eval_helpers = crossmodal.eval_helpers
eval_helpers.configure(buddy=buddy, trajectories=eval_trajectories, task="push")
eval_results = eval_helpers.run_eval()

# Save eval results
if args.save:
    buddy.add_metadata({"eval_results": eval_results})
