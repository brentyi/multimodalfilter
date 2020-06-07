from typing import Dict as _Dict

import diffbayes as _diffbayes
from .lstm import DoorLSTMFilter
from .pf import DoorParticleFilter, DoorMeasurementModel
from .dynamics import DoorDynamicsModel

model_types: _Dict[str, _diffbayes.base.Filter] = {}
""" (dict) Map from estimator model names to estimator model classes.
"""

for _model in (DoorLSTMFilter, DoorParticleFilter):
    model_types[_model.__name__] = _model
