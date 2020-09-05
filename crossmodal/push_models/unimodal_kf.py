import diffbayes
import diffbayes.types as types
import torch
import torch.nn as nn
from fannypack.nn import resblocks

from ..base_models import UnimodalKalmanFilter, UnimodalKalmanFilterMeasurementModel
from ..tasks import PushTask
from . import layers
from .dynamics import PushDynamicsModel
from .kf import PushKalmanFilter, PushKalmanFilterMeasurementModel


class PushUnimodalKalmanFilter(UnimodalKalmanFilter, PushTask.Filter):
    def __init__(self):
        """Initializes a kalman filter for our push task.
        """

        super().__init__(
            filter_models=[
                PushKalmanFilter(
                    dynamics_model=PushDynamicsModel(),
                    measurement_model=PushKalmanFilterMeasurementModel(
                        modalities={"image"}
                    ),
                ),
                PushKalmanFilter(
                    dynamics_model=PushDynamicsModel(),
                    measurement_model=PushKalmanFilterMeasurementModel(
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
            measurement_model=UnimodalKalmanFilterMeasurementModel(
                measurement_models=[
                    PushKalmanFilterMeasurementModel(modalities={"image"}),
                    PushKalmanFilterMeasurementModel(modalities={"pos", "sensors"}),
                ],
            ),
        )
