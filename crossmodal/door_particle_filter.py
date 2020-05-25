from typing import cast

import torch
import torch.nn as nn

import diffbayes
import diffbayes.types as types
from fannypack.nn import resblocks


class DoorParticleFilter(diffbayes.base.ParticleFilter):
    def __init__(self, num_particles=100):
        """Initializes a particle filter for our door task.
        """

        super().__init__(
            dynamics_model=DoorDynamicsModel(),
            measurement_model=DoorMeasurementModel(),
            num_particles=num_particles,
        )


class DoorDynamicsModel(diffbayes.base.DynamicsModel):
    def __init__(self, units=64):
        """Initializes a dynamics model for our door task.
        """

        Q = torch.diag([0.05, 0.05, 0.05])
        super().__init__(state_dim=3, Q=Q)

        control_dim = 7

        # Build neural network
        self.state_layers = nn.Sequential(
            nn.Linear(state_dim, units),
            resblocks.Linear(units),
            resblocks.Linear(units),
        )
        self.control_layers = nn.Sequential(
            nn.Linear(control_dim, units),
            resblocks.Linear(units),
            resblocks.Linear(units),
        )
        self.shared_layers = nn.Sequential(
            nn.Linear(units * 2, units),
            resblocks.Linear(units),
            resblocks.Linear(units),
            resblocks.Linear(units),
            nn.Linear(units, state_dim + 1),
        )
        self.units = units

    def forward(
        self,
        *,
        initial_states: types.StatesTorch,
        controls: types.ControlsTorch,
        noisy: bool,
    ) -> types.StatesTorch:
        """Dynamics forward pass for our door task.
        """
        N, state_dim = initial_states.shape
        assert state_dim == self.state_dim

        # (N, control_dim) => (N, units // 2)
        control_features = self.control_layers(controls)

        # (N, state_dim) => (N, units // 2)
        state_features = self.state_layers(initial_states)
        assert state_features.shape == (N, self.units)

        # (N, units)
        merged_features = torch.cat((control_features, state_features), dim=-1)
        assert merged_features.shape == (N, self.units * 2)

        # (N, units * 2) => (N, state_dim + 1)
        output_features = self.shared_layers(merged_features)
        assert output_features.shape == (N, state_dim + 1)

        # We separately compute a direction for our network and a "gate"
        # These are multiplied to produce our final state output
        state_update_direction = output_features[:, :state_dim]
        state_update_gate = torch.sigmoid(output_features[:, -1:])
        state_update = state_update_direction * state_update_gate
        assert state_update.shape == (N, state_dim)

        # Residual-style state update
        states_new = initial_states + state_update

        # Add noise
        self.add_noise(states=states_new, enabled=noisy)

        # Return (N, state_dim)
        return states_new


class DoorMeasurementModel(diffbayes.base.ParticleFilterMeasurementModel):
    def __init__(self, units=64):
        """Initializes a measurement model for our door task.
        """

        super().__init__(state_dim=3)
        obs_pos_dim = 3
        obs_sensors_dim = 7

        self.observation_image_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            resblocks.Conv2d(channels=32, kernel_size=3),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.Flatten(),  # 32 * 32 * 8
            nn.Linear(8 * 32 * 32, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
        )
        self.observation_pos_layers = nn.Sequential(
            nn.Linear(obs_pos_dim, units), resblocks.Linear(units),
        )
        self.observation_sensors_layers = nn.Sequential(
            nn.Linear(obs_sensors_dim, units), resblocks.Linear(units),
        )
        self.state_layers = nn.Sequential(nn.Linear(self.state_dim, units))

        self.shared_layers = nn.Sequential(
            nn.Linear(units * 4, units),
            nn.ReLU(inplace=True),
            resblocks.Linear(units),
            resblocks.Linear(units),
            nn.Linear(units, 1),
            # nn.LogSigmoid()
        )

        self.units = units

    def forward(
        self, *, states: types.StatesTorch, observations: types.ObservationsTorch
    ) -> types.StatesTorch:
        """Measurement forward pass for our door task.
        """
        assert type(observations) == dict
        assert len(states.shape) == 3  # (N, M, state_dim)
        assert states.shape[2] == self.state_dim
        observations = cast(types.TorchDict, observations)

        # N := distinct trajectory count
        # M := particle count
        N, M, _ = states.shape

        # Construct observations feature vector
        # (N, obs_dim)
        obs = []
        obs.append(self.observation_image_layers(observations["image"][:, None, :, :]))
        obs.append(self.observation_pos_layers(observations["gripper_pos"]))
        obs.append(self.observation_sensors_layers(observations["gripper_sensors"]))
        observation_features = torch.cat(obs, dim=1)

        # (N, obs_features) => (N, M, obs_features)
        observation_features = observation_features[:, None, :].expand(
            N, M, self.units * len(obs)
        )
        assert observation_features.shape == (N, M, self.units * len(obs))

        # (N, M, state_dim) => (N, M, units)
        state_features = self.state_layers(states)
        # state_features = self.state_layers(states * torch.tensor([[[1., 0.]]], device=states.device))
        assert state_features.shape == (N, M, self.units)

        merged_features = torch.cat((observation_features, state_features), dim=2)
        assert merged_features.shape == (N, M, self.units * (len(obs) + 1))

        # (N, M, merged_dim) => (N, M, 1)
        log_likelihoods = self.shared_layers(merged_features)
        assert log_likelihoods.shape == (N, M, 1)

        # Return (N, M)
        return torch.squeeze(log_likelihoods, dim=2)
