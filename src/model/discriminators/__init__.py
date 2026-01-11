from src.model.discriminators.CoMBD import CoMBD
from src.model.discriminators.SBD import SBD
from src.model.discriminators.discriminator_s import MultiScaleDiscriminator
from src.model.discriminators.discriminator_p import MultiPeriodDiscriminator
from src.model.discriminators.ms_stft import MultiScaleSTFTDiscriminator

__all__ = [
    "MultiScaleDiscriminator",
    "MultiPeriodDiscriminator",
    "CoMBD",
    "SBD",
    "MultiScaleSTFTDiscriminator"
]