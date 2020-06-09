from typing import Dict as _Dict

import diffbayes as _diffbayes

from .crossmodal_pf import DoorCrossmodalParticleFilter, DoorCrossmodalWeightModel
from .dynamics import DoorDynamicsModel
from .lstm import DoorLSTMFilter
from .pf import DoorMeasurementModel, DoorParticleFilter

model_types: _Dict[str, _diffbayes.base.Filter] = {}
""" (dict) Map from estimator model names to estimator model classes.
"""

for _model in (DoorLSTMFilter, DoorParticleFilter, DoorCrossmodalParticleFilter):
    model_types[_model.__name__] = _model
