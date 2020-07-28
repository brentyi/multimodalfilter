import diffbayes

from ..base_models import CrossmodalParticleFilterMeasurementModel
from ..tasks import PushTask
from .dynamics import PushDynamicsModel
from .pf import PushMeasurementModel


class PushUnimodalParticleFilter(diffbayes.base.ParticleFilter, PushTask.Filter):
    def __init__(self):
        """Initializes a particle filter for our door task.
        """

        super().__init__(
            dynamics_model=PushDynamicsModel(),
            measurement_model=CrossmodalParticleFilterMeasurementModel(
                measurement_models=[
                    PushMeasurementModel(modalities={"image"}),
                    PushMeasurementModel(modalities={"pos", "sensors"}),
                ],
                crossmodal_weight_model=None,
                state_dim=2,
            ),
            num_particles=30,
        )

    def train(self, mode: bool = True):
        """Adjust particle count based on train vs eval mode.
        """
        self.num_particles = 30 if mode else 300
        super().train(mode)
