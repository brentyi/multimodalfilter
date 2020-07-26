import torch
import torch.nn as nn

import diffbayes
import diffbayes.types as types
from fannypack.nn import resblocks

from ..base_models import (
    UnimodalKalmanFilter,
    UnimodalKalmanFilterMeasurementModel,
)
from . import layers
from .dynamics import PushDynamicsModel
from .kf import PushKalmanFilterMeasurementModel
from .kf import PushKalmanFilter

class PushUnimodalKalmanFilter(UnimodalKalmanFilter):
    def __init__(self):
        """Initializes a kalman filter for our push task.
        """

        super().__init__(
            filter_models= [
                PushKalmanFilter(
                    dynamics_model=PushDynamicsModel(),
                    measurement_model=PushKalmanFilterMeasurementModel(
                        modalities={"image", "pos"}
                    ),

                ),
                PushKalmanFilter(
                    dynamics_model=PushDynamicsModel(),
                    measurement_model=PushKalmanFilterMeasurementModel(
                        modalities={"pos", "sensors"}
                    ),
                )
            ],
            state_dim=2,
        )

class PushMeasurementUnimodalKalmanFilter(PushKalmanFilter):
    def __init__(self):
        """Initializes a kalman filter for our push task.
        """

        super().__init__(
            dynamics_model=PushDynamicsModel(),
            measurement_model=UnimodalKalmanFilterMeasurementModel(
                measurement_models=[
                    PushKalmanFilterMeasurementModel(modalities={"image"}),
                    PushKalmanFilterMeasurementModel(modalities={"pos", "sensors"})
                ],
            ),
        )