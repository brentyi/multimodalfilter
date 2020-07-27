from typing import Dict as _Dict

import diffbayes as _diffbayes

from .crossmodal_kf import (
    PushCrossmodalKalmanFilter,
    PushCrossmodalKalmanFilterWeightModel,
    PushKalmanFilterMeasurementModel,
    PushMeasurementCrossmodalKalmanFilter,
)
from .crossmodal_pf import PushCrossmodalParticleFilter, PushCrossmodalWeightModel
from .dynamics import PushDynamicsModel
from .kf import PushKalmanFilter, PushKalmanFilterMeasurementModel
from .lstm import PushLSTMFilter
from .pf import PushMeasurementModel, PushParticleFilter
from .unimodal_kf import PushMeasurementUnimodalKalmanFilter, PushUnimodalKalmanFilter
from .unimodal_pf import PushUnimodalParticleFilter

model_types: _Dict[str, _diffbayes.base.Filter] = {}
""" (dict) Map from estimator model names to estimator model classes.
"""

# TODO: this would be cleaner if we registered model types with a decorator
for _model in (
    PushLSTMFilter,
    PushParticleFilter,
    PushCrossmodalParticleFilter,
    PushUnimodalParticleFilter,
    PushKalmanFilter,
    PushCrossmodalKalmanFilter,
    PushUnimodalKalmanFilter,
    PushMeasurementCrossmodalKalmanFilter,
    PushMeasurementUnimodalKalmanFilter,
):
    model_types[_model.__name__] = _model
