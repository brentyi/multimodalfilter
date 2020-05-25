import numpy as np
import torch.utils.data

import crossmodal
import diffbayes
import fannypack

# Open PDB before quitting
fannypack.utils.pdb_safety_net()

# Create model, Buddy
filter_model = crossmodal.door_particle_filter.DoorParticleFilter()
buddy = fannypack.utils.Buddy("pf-test1", filter_model)

# Load training data
trajectories = crossmodal.door_data.load_trajectories("panda_door_pull_10.hdf5")


def train_dynamics_single_step():
    dataloader = torch.utils.data.DataLoader(
        diffbayes.data.SingleStepDataset(trajectories=trajectories),
        batch_size=32,
        shuffle=True,
        num_workers=8,
    )
    diffbayes.train.train_dynamics_single_step(
        buddy, filter_model.dynamics_model, dataloader
    )


# train_dynamics_single_step()


def train_dynamics_recurrent(subsequence_length):
    dataloader = torch.utils.data.DataLoader(
        diffbayes.data.SubsequenceDataset(
            trajectories=trajectories, subsequence_length=subsequence_length
        ),
        batch_size=32,
        shuffle=True,
        num_workers=8,
    )
    diffbayes.train.train_dynamics_recurrent(
        buddy, filter_model.dynamics_model, dataloader
    )


train_dynamics_recurrent(5)


def train_measurement():
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
    diffbayes.train.train_particle_filter_measurement_model(
        buddy, filter_model.measurement_model, dataloader
    )


def train_e2e(subsequence_length):
    dataloader = torch.utils.data.DataLoader(
        diffbayes.data.SubsequenceDataset(
            trajectories=trajectories, subsequence_length=subsequence_length
        ),
        batch_size=32,
        shuffle=True,
        num_workers=8,
    )
    diffbayes.train.train_filter(
        buddy, filter_model, dataloader, initial_covariance=torch.eye(3)
    )


train_measurement()


train_e2e(subsequence_length=3)
train_e2e(subsequence_length=10)
train_e2e(subsequence_length=20)
