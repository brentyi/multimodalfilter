from typing import Dict as _Dict

import diffbayes as _diffbayes

from .crossmodal_pf import DoorCrossmodalParticleFilter
    # DoorCrossmodalWeightModel
from .dynamics import DoorDynamicsModel
from .lstm import DoorLSTMFilter
from .pf import DoorMeasurementModel, DoorParticleFilter
from .kf import DoorKalmanFilterMeasurementModel, DoorKalmanFilter
from .crossmodal_kf import DoorCrossmodalKalmanFilter, DoorMeasurementCrossmodalKalmanFilter, \
    DoorCrossmodalKalmanFilterWeightModel
from .unimodal_kf import DoorUnimodalKalmanFilter, DoorMeasurementUnimodalKalmanFilter

model_types: _Dict[str, _diffbayes.base.Filter] = {}
""" (dict) Map from estimator model names to estimator model classes.
"""

for _model in (DoorLSTMFilter,
               DoorParticleFilter, DoorCrossmodalParticleFilter,
               DoorCrossmodalKalmanFilter,
               DoorKalmanFilter, DoorMeasurementCrossmodalKalmanFilter,
               DoorUnimodalKalmanFilter, DoorMeasurementUnimodalKalmanFilter):
    model_types[_model.__name__] = _model
