import diffbayes

from ..base_models import CrossmodalParticleFilterMeasurementModel
from ..tasks import DoorTask
from .dynamics import DoorDynamicsModel
from .pf import DoorMeasurementModel


class DoorUnimodalParticleFilter(diffbayes.base.ParticleFilter, DoorTask.Filter):
    def __init__(self):
        """Initializes a particle filter for our door task.
        """

        super().__init__(
            dynamics_model=DoorDynamicsModel(),
            measurement_model=CrossmodalParticleFilterMeasurementModel(
                measurement_models=[
                    DoorMeasurementModel(modalities={"image"}),
                    DoorMeasurementModel(modalities={"pos", "sensors"}),
                ],
                crossmodal_weight_model=None,
                state_dim=3,
            ),
            num_particles=30,
        )

    def train(self, mode: bool = True):
        """Adjust particle count based on train vs eval mode.
        """
        self.num_particles = 30 if mode else 300
        super().train(mode)
