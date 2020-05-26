import argparse

import crossmodal
import diffbayes
import fannypack

# Move cache in case we're running on NFS (eg Juno), open PDB on quit
fannypack.data.set_cache_path(crossmodal.__path__[0] + "/../.cache")
fannypack.utils.pdb_safety_net()

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str)
args = parser.parse_args()

# Create and validate Buddy
buddy = fannypack.utils.Buddy(args.experiment_name)
assert "model_type" in buddy.metadata
assert "dataset_args" in buddy.metadata

# Load model using experiment metadata
filter_model: diffbayes.base.Filter = {
    "lstm": crossmodal.door_lstm.DoorLSTMFilter,
    "particle_filter": crossmodal.door_particle_filter.DoorParticleFilter,
}[buddy.metadata["model_type"]]()
buddy.attach_model(filter_model)
buddy.load_checkpoint()

# Run eval
crossmodal.door_eval.eval_model(filter_model, buddy.metadata["dataset_args"])
