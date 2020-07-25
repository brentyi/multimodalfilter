from typing import Dict as _Dict

import diffbayes as _diffbayes

from .crossmodal_pf import PushCrossmodalParticleFilter, PushCrossmodalWeightModel
from .dynamics import PushDynamicsModel
from .lstm import PushLSTMFilter
from .pf import PushMeasurementModel, PushParticleFilter
from .kf import PushKalmanFilter, PushKalmanFilterMeasurementModel
from .crossmodal_kf import PushCrossmodalKalmanFilter, PushCrossmodalKalmanFilterWeightModel, \
    PushMeasurementCrossmodalKalmanFilter, PushKalmanFilterMeasurementModel
from .unimodal_kf import PushUnimodalKalmanFilter, PushMeasurementUnimodalKalmanFilter

model_types: _Dict[str, _diffbayes.base.Filter] = {}
""" (dict) Map from estimator model names to estimator model classes.
"""

for _model in (PushLSTMFilter, PushParticleFilter, PushCrossmodalParticleFilter,
               PushKalmanFilter, PushCrossmodalKalmanFilter, PushUnimodalKalmanFilter,
               PushMeasurementCrossmodalKalmanFilter, PushMeasurementUnimodalKalmanFilter):
    model_types[_model.__name__] = _model
