from typing import cast

import torch
import torch.nn as nn

import diffbayes
import diffbayes.types as types
from fannypack.nn import resblocks

from . import layers


class DoorLSTMFilter(diffbayes.base.Filter):
    def __init__(self, units: int = 64):
        """Initializes an LSTM architecture for our door task.
        """
        super().__init__(state_dim=3)

        self.lstm_hidden_dim = 4
        self.lstm_num_layers = 2
        self.units = units

        # Observation encoders
        self.image_rows = 32
        self.image_cols = 32
        self.observation_image_layers = layers.observation_image_layers(units)
        self.observation_pos_layers = layers.observation_pos_layers(units)
        self.observation_sensors_layers = layers.observation_sensors_layers(units)

        # Control layers
        self.control_layers = layers.control_layers(units)

        # Initial fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(units * 4, units), nn.ReLU(inplace=True), resblocks.Linear(units),
        )

        # LSTM layers
        self.lstm = nn.LSTM(units, self.lstm_hidden_dim, self.lstm_num_layers)

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(self.lstm_hidden_dim, units),
            nn.ReLU(inplace=True),
            nn.Linear(units, self.state_dim),
        )

    def initialize_beliefs(
        self, *, mean: types.StatesTorch, covariance: torch.Tensor
    ) -> None:
        N = len(mean)
        device = next(self.parameters()).device
        self.lstm_hidden = (
            torch.zeros(self.lstm_num_layers, N, self.lstm_hidden_dim, device=device),
            torch.zeros(self.lstm_num_layers, N, self.lstm_hidden_dim, device=device),
        )

    def forward_loop(
        self, *, observations: types.ObservationsTorch, controls: types.ControlsTorch
    ) -> types.StatesTorch:
        observations = cast(types.TorchDict, observations)

        # Observations: key->value
        # where shape of value is (seq_len, batch, *)
        T, N = observations["image"].shape[:2]
        assert observations["gripper_pos"].shape[:2] == (T, N)
        assert observations["gripper_sensors"].shape[:2] == (T, N)

        # Forward pass through observation encoders
        reshaped_images = observations["image"].reshape(
            T * N, 1, self.image_rows, self.image_cols
        )
        image_features = self.observation_image_layers(reshaped_images).reshape(
            (T, N, self.units)
        )

        merged_features = torch.cat(
            (
                image_features,
                self.observation_pos_layers(observations["gripper_pos"]),
                self.observation_sensors_layers(observations["gripper_sensors"]),
                self.control_layers(controls),
            ),
            dim=-1,
        )

        assert merged_features.shape == (T, N, self.units * 4)

        fused_features = self.fusion_layers(merged_features)
        assert fused_features.shape == (T, N, self.units)

        # Forward pass through LSTM layer + save the hidden state
        lstm_out, self.lstm_hidden = self.lstm(fused_features, self.lstm_hidden)
        assert lstm_out.shape == (T, N, self.lstm_hidden_dim)

        predicted_states = self.output_layers(lstm_out)
        assert predicted_states.shape == (T, N, self.state_dim)

        return predicted_states
