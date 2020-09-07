import torch
import torch.nn as nn
import torchfilter
import torchfilter.types as types
from fannypack.nn import resblocks

from ..base_models import UnimodalKalmanFilter, UnimodalVirtualSensorModel
from ..tasks import PushTask
from . import layers
from .dynamics import PushDynamicsModel
from .kf import PushKalmanFilter, PushVirtualSensorModel


class PushUnimodalKalmanFilter(UnimodalKalmanFilter, PushTask.Filter):
    def __init__(self):
        """Initializes a kalman filter for our push task.
        """

        super().__init__(
            filter_models=[
                PushKalmanFilter(
                    dynamics_model=PushDynamicsModel(),
                    virtual_sensor_model=PushVirtualSensorModel(modalities={"image"}),
                ),
                PushKalmanFilter(
                    dynamics_model=PushDynamicsModel(),
                    virtual_sensor_model=PushVirtualSensorModel(
                        modalities={"pos", "sensors"}
                    ),
                ),
            ],
            state_dim=2,
        )


class PushMeasurementUnimodalKalmanFilter(PushKalmanFilter):
    def __init__(self):
        """Initializes a kalman filter for our push task.
        """

        super().__init__(
            dynamics_model=PushDynamicsModel(),
            virtual_sensor_model=UnimodalVirtualSensorModel(
                virtual_sensor_model=[
                    PushVirtualSensorModel(modalities={"image"}),
                    PushVirtualSensorModel(modalities={"pos", "sensors"}),
                ],
            ),
        )
