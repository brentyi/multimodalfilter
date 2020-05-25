import numpy as np
import torch
import torch.utils.data

import crossmodal
import diffbayes
import fannypack

# Open PDB before quitting
fannypack.utils.pdb_safety_net()

# Load training data
trajectories = crossmodal.door_data.load_trajectories(
    "panda_door_pull_300.hdf5", "panda_door_push_300.hdf5"
)

# Create model, Buddy
filter_model = crossmodal.door_lstm.DoorLSTMFilter()
buddy = fannypack.utils.Buddy("lstm-test1", filter_model)

# Training helper
def train(*, subsequence_length, batch_size, epochs):
    dataloader = torch.utils.data.DataLoader(
        diffbayes.data.SubsequenceDataset(
            trajectories=trajectories, subsequence_length=subsequence_length
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    for _ in range(epochs):
        diffbayes.train.train_filter(
            buddy, filter_model, dataloader, initial_covariance=torch.eye(3)
        )

# Run training curriculum
train(subsequence_length=2, batch_size=32, epochs=2)
buddy.save_checkpoint("phase0")

train(subsequence_length=16, batch_size=32, epochs=20)
buddy.save_checkpoint("phase1")
