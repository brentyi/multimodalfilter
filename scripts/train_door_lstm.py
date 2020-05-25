import numpy as np
import torch
import torch.utils.data

import crossmodal
import diffbayes
import fannypack

# Create model, Buddy
filter_model = crossmodal.door_lstm.DoorLSTMFilter()
buddy = fannypack.utils.Buddy("lstm-test1", filter_model)


def train(*, subsequence_length, batch_size, epochs):
    trajectories = crossmodal.door_data.load_trajectories("panda_door_pull_10.hdf5")
    dataloader = torch.utils.data.DataLoader(
        diffbayes.data.SubsequenceDataset(
            trajectories=trajectories, subsequence_length=subsequence_length
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=8
    )

    for _ in range(epochs):
        diffbayes.train.train_filter(
            buddy, filter_model, dataloader, initial_covariance=torch.eye(3)
        )


train(subsequence_length=3, batch_size=32, epochs=5)
train(subsequence_length=10, batch_size=32, epochs=5)
train(subsequence_length=20, batch_size=32, epochs=5)
