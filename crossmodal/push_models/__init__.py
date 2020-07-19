from typing import Dict as _Dict

import diffbayes as _diffbayes

from .crossmodal_pf import PushCrossmodalParticleFilter, PushCrossmodalWeightModel
from .dynamics import PushDynamicsModel
from .lstm import PushLSTMFilter
from .pf import PushMeasurementModel, PushParticleFilter

model_types: _Dict[str, _diffbayes.base.Filter] = {}
""" (dict) Map from estimator model names to estimator model classes.
"""

for _model in (PushLSTMFilter, PushParticleFilter, PushCrossmodalParticleFilter):
    model_types[_model.__name__] = _model
