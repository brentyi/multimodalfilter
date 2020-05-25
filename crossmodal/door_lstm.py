from typing import cast

import torch
import torch.nn as nn

import diffbayes
import diffbayes.types as types
from fannypack.nn import resblocks


class DoorLSTMFilter(diffbayes.base.Filter):
    def __init__(self, units=32):
        """Initializes an LSTM architecture for our door task.
        """
        super().__init__(state_dim=3)

        obs_pos_dim = 3
        obs_sensors_dim = 7
        control_dim = 7
        self.lstm_hidden_dim = 4
        self.lstm_num_layers = 2
        self.units = units

        # Observation encoders
        self.image_rows = 32
        self.image_cols = 32
        self.observation_image_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            resblocks.Conv2d(channels=32, kernel_size=3),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding=1),
            nn.Flatten(),  # 32 * 32 * 4
            nn.Linear(4 * 32 * 32, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
        )
        self.observation_pose_layers = nn.Sequential(
            nn.Linear(obs_pos_dim, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
        )
        self.observation_sensors_layers = nn.Sequential(
            nn.Linear(obs_sensors_dim, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
        )

        # Control layers
        self.control_layers = nn.Sequential(
            nn.Linear(control_dim, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
        )

        # Fusion layer
        self.fusion_layers = nn.Sequential(
            nn.Linear(units * 4, units), nn.ReLU(inplace=True), resblocks.Linear(units),
        )

        # LSTM layers
        self.lstm = nn.LSTM(
            units, self.lstm_hidden_dim, self.lstm_num_layers, batch_first=True
        )

        # Define the output layer
        self.output_layers = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, units),
            nn.ReLU(inplace=True),
            nn.Linear(units, self.state_dim),
        )

    def initialize_beliefs(
        self, *, mean: types.StatesTorch, covariance: torch.Tensor
    ) -> None:
        """Initialize filter beliefs. This doesn't make sense for our LSTM estimator, so
        we do nothing. :)
        """
        N = len(mean)
        device = next(self.parameters()).device
        self.lstm_hidden = (
            torch.zeros(self.lstm_num_layers, N, self.lstm_hidden_dim),
            torch.zeros(self.lstm_num_layers, N, self.lstm_hidden_dim),
        )

    def forward_loop(
        self, *, observations: types.ObservationsTorch, controls: types.ControlsTorch
    ) -> types.StatesTorch:
        """LSTM "looped" forward pass.
        """
        observations = cast(types.TorchDict, observations)

        # Observations: key->value
        # where shape of value is (batch, seq_len, *)
        batch_size = observations["image"].shape[0]
        sequence_length = observations["image"].shape[1]
        assert observations["image"].shape[0] == batch_size
        assert observations["gripper_pos"].shape[0] == batch_size
        assert observations["gripper_pos"].shape[1] == sequence_length
        assert observations["gripper_sensors"].shape[1] == sequence_length

        # Forward pass through observation encoders
        reshaped_images = observations["image"].reshape(
            batch_size * sequence_length, 1, self.image_rows, self.image_cols
        )
        image_features = self.observation_image_layers(reshaped_images).reshape(
            (batch_size, sequence_length, self.units)
        )

        merged_features = torch.cat(
            (
                image_features,
                self.observation_pose_layers(observations["gripper_pos"]),
                self.observation_sensors_layers(observations["gripper_sensors"]),
                self.control_layers(controls),
            ),
            dim=-1,
        )

        assert merged_features.shape == (batch_size, sequence_length, self.units * 4)

        fused_features = self.fusion_layers(merged_features)
        assert fused_features.shape == (batch_size, sequence_length, self.units)

        # Forward pass through LSTM layer + save the hidden state
        lstm_out, self.lstm_hidden = self.lstm(fused_features, self.lstm_hidden)
        assert lstm_out.shape == (batch_size, sequence_length, self.lstm_hidden_dim)

        predicted_states = self.output_layers(lstm_out)
        assert predicted_states.shape == (batch_size, sequence_length, self.state_dim)

        return predicted_states
