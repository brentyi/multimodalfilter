import argparse
import dataclasses

import crossmodal
import diffbayes
import fannypack

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

# Load trajectories using experiment metadata
eval_trajectories = Task.get_eval_trajectories(**dataset_args)

states = []
observations = fannypack.utils.SliceWrapper({})
controls = []
for traj in eval_trajectories:
    states.append(traj[0])
    observations.append(traj[1])
    controls.append(traj[2])
controls = np.stack([traj[2] for traj in eval_trajectories], axis=0)

# Run eval
eval_helpers = crossmodal.eval_helpers
eval_helpers.configure(buddy=buddy, trajectories=eval_trajectories)
eval_results = eval_helpers.run_eval()

# Save eval results
if args.save:
    buddy.add_metadata({"eval_results": dataclasses.asdict(eval_results)})
