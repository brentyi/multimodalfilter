from typing import Dict as _Dict

import diffbayes as _diffbayes

from .crossmodal_kf import (
    DoorCrossmodalKalmanFilter,
    DoorCrossmodalKalmanFilterWeightModel,
    DoorMeasurementCrossmodalKalmanFilter,
)
from .crossmodal_pf import DoorCrossmodalParticleFilter
from .unimodal_pf import DoorUnimodalParticleFilter
from .dynamics import DoorDynamicsModel
from .kf import DoorKalmanFilter, DoorKalmanFilterMeasurementModel
from .lstm import DoorLSTMFilter
from .pf import DoorMeasurementModel, DoorParticleFilter
from .unimodal_kf import DoorMeasurementUnimodalKalmanFilter, DoorUnimodalKalmanFilter

# DoorCrossmodalWeightModel

model_types: _Dict[str, _diffbayes.base.Filter] = {}
""" (dict) Map from estimator model names to estimator model classes.
"""

for _model in (
    DoorLSTMFilter,
    DoorParticleFilter,
    DoorCrossmodalParticleFilter,
    DoorUnimodalParticleFilter,
    DoorKalmanFilter,
    DoorCrossmodalKalmanFilter,
    DoorMeasurementCrossmodalKalmanFilter,
    DoorUnimodalKalmanFilter,
    DoorMeasurementUnimodalKalmanFilter,
):
    model_types[_model.__name__] = _model
