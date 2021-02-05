import torch
import torch.nn as nn
import torchfilter
import torchfilter.types as types
from fannypack.nn import resblocks

from ..base_models import UnimodalKalmanFilter, UnimodalVirtualSensorModel
from ..tasks import DoorTask
from . import layers
from .dynamics import DoorDynamicsModel
from .kf import DoorKalmanFilter, DoorVirtualSensorModel


class DoorUnimodalKalmanFilter(UnimodalKalmanFilter, DoorTask.Filter):
    def __init__(self):
        """Initializes a kalman filter for our door task."""

        super().__init__(
            filter_models=[
                DoorKalmanFilter(
                    dynamics_model=DoorDynamicsModel(),
                    virtual_sensor_model=DoorVirtualSensorModel(modalities={"image"}),
                ),
                DoorKalmanFilter(
                    dynamics_model=DoorDynamicsModel(),
                    virtual_sensor_model=DoorVirtualSensorModel(
                        modalities={"pos", "sensors"}
                    ),
                ),
            ],
            state_dim=3,
        )


class DoorMeasurementUnimodalKalmanFilter(DoorKalmanFilter):
    def __init__(self):
        """Initializes a kalman filter for our door task."""

        super().__init__(
            dynamics_model=DoorDynamicsModel(),
            virtual_sensor_model=UnimodalVirtualSensorModel(
                virtual_sensor_model=[
                    DoorVirtualSensorModel(modalities={"image"}),
                    DoorVirtualSensorModel(modalities={"pos", "sensors"}),
                ],
                state_dim=3,
            ),
        )
