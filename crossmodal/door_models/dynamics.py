import torch
import torch.nn as nn

import diffbayes
import diffbayes.types as types
from fannypack.nn import resblocks

from . import layers


class DoorDynamicsModel(diffbayes.base.DynamicsModel):
    def __init__(self, units=64):
        """Initializes a dynamics model for our door task.
        """

        super().__init__(state_dim=3)

        control_dim = 7

        # Fixed dynamics covariance
        self.Q = torch.diag(torch.FloatTensor([0.05, 0.01, 0.01]))

        # Build neural network
        self.state_layers = layers.state_layers(units=units)
        self.control_layers = layers.control_layers(units=units)
        self.shared_layers = nn.Sequential(
            nn.Linear(units * 2, units),
            resblocks.Linear(units),
            resblocks.Linear(units),
            resblocks.Linear(units),
            nn.Linear(units, self.state_dim + 1),
        )
        self.units = units

    def forward(
        self, *, initial_states: types.StatesTorch, controls: types.ControlsTorch,
    ) -> types.StatesTorch:
        N, state_dim = initial_states.shape
        assert state_dim == self.state_dim

        # (N, control_dim) => (N, units // 2)
        control_features = self.control_layers(controls)

        # (N, state_dim) => (N, units // 2)
        state_features = self.state_layers(initial_states)
        assert state_features.shape == (N, self.units)

        # (N, units)
        merged_features = torch.cat((control_features, state_features), dim=1)
        assert merged_features.shape == (N, self.units * 2)

        # (N, units * 2) => (N, state_dim + 1)
        output_features = self.shared_layers(merged_features)
        assert output_features.shape == (N, state_dim + 1)

        # We separately compute a direction for our network and a scalar "gate"
        # These are multiplied to produce our final state output
        state_update_direction = output_features[:, :state_dim]
        state_update_gate = torch.sigmoid(output_features[:, -1:])
        state_update = state_update_direction * state_update_gate
        assert state_update.shape == (N, state_dim)

        # Return residual-style state update, constant uncertainties
        states_new = initial_states + state_update
        covariances = self.Q[None, :, :].expand(N, state_dim, state_dim)
        return states_new, covariances
