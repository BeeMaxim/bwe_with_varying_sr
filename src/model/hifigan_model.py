import torch.nn as nn
from src.model.generator import A2AHiFiPlusGenerator
from src.model.discriminator_p import MultiPeriodDiscriminator
from src.model.discriminator_s import MultiScaleDiscriminator

from hydra.utils import instantiate
import hydra


class HiFiGAN(nn.Module):
    def __init__(self,
                 generator_config,
                 discriminators):
        super().__init__()

        self.discriminators = nn.ModuleDict()
    
        for disc_name, disc in discriminators.items():
            self.discriminators[disc_name] = disc

        self.generator = A2AHiFiPlusGenerator(**generator_config)


    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        gen_parameters = sum(
            [p.numel() for p in self.generator.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"
        result_info = result_info + f"\nGen: {gen_parameters}"
        for disc_name, disc in self.discriminators.items():
            parameters = sum(
                [p.numel() for p in disc.parameters() if p.requires_grad]
            )
            result_info = result_info + f"\n{disc_name}: {parameters}"

        return result_info