import torch
import torch.nn as nn

import diffbayes
import diffbayes.types as types
from fannypack.nn import resblocks

from ..base_models import UnimodalKalmanFilter, UnimodalKalmanFilterMeasurementModel
from ..tasks import DoorTask
from . import layers
from .dynamics import DoorDynamicsModel
from .kf import DoorKalmanFilter, DoorKalmanFilterMeasurementModel


class DoorUnimodalKalmanFilter(UnimodalKalmanFilter, DoorTask.Filter):
    def __init__(self):
        """Initializes a kalman filter for our door task.
        """

        super().__init__(
            filter_models=[
                DoorKalmanFilter(
                    dynamics_model=DoorDynamicsModel(),
                    measurement_model=DoorKalmanFilterMeasurementModel(
                        modalities={"image"}
                    ),
                ),
                DoorKalmanFilter(
                    dynamics_model=DoorDynamicsModel(),
                    measurement_model=DoorKalmanFilterMeasurementModel(
                        modalities={"pos", "sensors"}
                    ),
                ),
            ],
            state_dim=3,
        )


class DoorMeasurementUnimodalKalmanFilter(DoorKalmanFilter):
    def __init__(self):
        """Initializes a kalman filter for our door task.
        """

        super().__init__(
            dynamics_model=DoorDynamicsModel(),
            measurement_model=UnimodalKalmanFilterMeasurementModel(
                measurement_models=[
                    DoorKalmanFilterMeasurementModel(modalities={"image"}),
                    DoorKalmanFilterMeasurementModel(modalities={"pos", "sensors"}),
                ],
                state_dim=3,
            ),
        )
