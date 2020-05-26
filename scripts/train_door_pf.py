import argparse

import numpy as np
import torch.utils.data

import crossmodal
import diffbayes
import fannypack

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--experiment_name", type=str)
parser.add_argument("--freeze_dynamics", action="store_true")
args = parser.parse_args()

# Move cache in case we're running on NFS (eg Juno), open PDB on quit
fannypack.data.set_cache_path(crossmodal.__path__[0] + "/../.cache")
fannypack.utils.pdb_safety_net()

# Load training data
trajectories = crossmodal.door_data.load_trajectories(
    "panda_door_pull_300.hdf5", "panda_door_push_300.hdf5"
)

# Create model, Buddy
filter_model = crossmodal.door_particle_filter.DoorParticleFilter()
buddy = fannypack.utils.Buddy(args.experiment_name, filter_model)
buddy.set_metadata({"model_type": "particle_filter", "dataset_args": {}})

# Training helpers
def train_dynamics_single_step(*, epochs):
    dataloader = torch.utils.data.DataLoader(
        diffbayes.data.SingleStepDataset(trajectories=trajectories),
        batch_size=32,
        shuffle=True,
        num_workers=8,
    )
    for _ in range(epochs):
        diffbayes.train.train_dynamics_single_step(
            buddy, filter_model.dynamics_model, dataloader
        )


def train_dynamics_recurrent(*, subsequence_length, epochs):
    dataloader = torch.utils.data.DataLoader(
        diffbayes.data.SubsequenceDataset(
            trajectories=trajectories, subsequence_length=subsequence_length
        ),
        batch_size=32,
        shuffle=True,
        num_workers=8,
    )
    for _ in range(epochs):
        diffbayes.train.train_dynamics_recurrent(
            buddy, filter_model.dynamics_model, dataloader
        )


def train_measurement(*, epochs):
    dataloader = torch.utils.data.DataLoader(
        diffbayes.data.ParticleFilterMeasurementDataset(
            trajectories=trajectories,
            covariance=np.diag([0.2, 0.2, 0.2]),
            samples_per_pair=10,
        ),
        batch_size=64,
        shuffle=True,
        num_workers=8,
    )
    for _ in range(epochs):
        diffbayes.train.train_particle_filter_measurement_model(
            buddy, filter_model.measurement_model, dataloader
        )


def train_e2e(*, subsequence_length, epochs):
    dataloader = torch.utils.data.DataLoader(
        diffbayes.data.SubsequenceDataset(
            trajectories=trajectories, subsequence_length=subsequence_length
        ),
        batch_size=32,
        shuffle=True,
        num_workers=8,
    )
    for _ in range(epochs):
        diffbayes.train.train_filter(
            buddy,
            filter_model,
            dataloader,
            initial_covariance=torch.eye(3, device=buddy.device) * 0.1,
        )


train_dynamics_single_step(epochs=5)
buddy.save_checkpoint("phase0")

train_dynamics_recurrent(subsequence_length=10, epochs=5)
train_dynamics_recurrent(subsequence_length=20, epochs=5)
buddy.save_checkpoint("phase1")

train_measurement(epochs=5)
buddy.save_checkpoint("phase2")

if args.freeze_dynamics:
    fannypack.utils.freeze_module(filter_model.dynamics_model)

train_e2e(subsequence_length=3, epochs=5)
train_e2e(subsequence_length=10, epochs=5)
train_e2e(subsequence_length=20, epochs=5)
buddy.save_checkpoint("phase3")
