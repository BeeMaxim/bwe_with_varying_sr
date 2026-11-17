import torch 
import torch.nn as nn
import torch.nn.functional as F
from src.loss.msstft_loss import MultiResolutionSTFTLoss


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, disc_gt_output, disc_predicted_output):
        total_gt_loss = 0
        total_pred_loss = 0
        for gt_output, pred_output in zip(disc_gt_output, disc_predicted_output):
            gt_loss = torch.mean((1 - gt_output)**2)
            pred_loss = torch.mean(pred_output**2)
            total_gt_loss += gt_loss
            total_pred_loss += pred_loss

        return total_gt_loss, total_pred_loss

        
class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dsc_output):
        loss = 0.0
        for predicted in dsc_output:
            pred_loss = torch.mean((1 - predicted)**2)
            loss += pred_loss
        return loss
    

class FeatureMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, initial, predicted):
        loss = 0
        for disc_initial_feat, disc_pred_feat in zip(initial, predicted):
            for initial_feat, predicted_feat in zip(disc_initial_feat, disc_pred_feat):
                loss += torch.mean(torch.abs(initial_feat - predicted_feat))
        return loss     


class MelSpectrogramLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, initial_spec, pred_spec):
        return F.l1_loss(pred_spec, initial_spec)
   

class SpectrogramLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, initial_spec, pred_spec):
        return F.l1_loss(pred_spec, initial_spec)

  
class HiFiGANLoss(nn.Module):
    def __init__(self, 
                 loss_type="msstft",
                 spectral_complex_loss=True):
        super().__init__()
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()

        if loss_type == "melspec":
            self.melspec_loss = MelSpectrogramLoss()
        elif loss_type == "msstft":
            self.msstft_loss = MultiResolutionSTFTLoss(spectral_complex_loss=spectral_complex_loss)
        else:
            raise Exception("Unknown loss parameter")

        self.loss_type = loss_type

        self.fm_loss = FeatureMatchingLoss()
              
    def discriminator_loss(self, batch):
        total_loss = 0
        losses = {}

        for key in batch:
            if key.endswith("gt_out"):
                disc_name = key[:-7]
                disc_loss = self.disc_loss(batch[key], batch[f"{disc_name}_fake_out"])
                losses[f"{disc_name}_disc_loss"] = disc_loss
                total_loss = total_loss + disc_loss

        return losses, total_loss
        
    def generator_loss(self, batch):
        adv_losses = {}
        feats_losses = {}
        total_loss = 0

        # TODO computation of mel specs here with given melSpecComputer as an argument
        # for better generalization to other spectral losses
        # gen_hr_loss, fake_gen_loss, gen_loss = self.criterion.generator_loss(batch)
        if self.loss_type == "msstft":
            hr_spec_loss = self.msstft_loss(batch["generated_hr"].squeeze(1), batch["wav_hr"].squeeze(1))
            fake_spec_loss = self.msstft_loss(batch["generated_wav"].squeeze(1), batch["wav_hr"].squeeze(1))
        elif self.loss_type == "melspec":
            spec_loss = self.melspec_loss(batch["mel_spec_hr"], batch["mel_spec_fake"])

        total_loss = 45 * fake_spec_loss + 45 * hr_spec_loss
        
        return hr_spec_loss, fake_spec_loss, total_loss

    def exploration_adversarial_loss(self, gt_feats, fake_out, fake_feats):
        adv_gen_loss = self.gen_loss(fake_out)
        feats_gen_loss = self.fm_loss(gt_feats, fake_feats)
        return adv_gen_loss, feats_gen_loss, adv_gen_loss + 2 * feats_gen_loss

    def exploration_reconstruction_loss(self, wav_fake, target_wav):
        return 45 * self.msstft_loss(wav_fake.squeeze(1), target_wav.squeeze(1))

    def exploration_discriminator_loss(self, gt_out, fake_out):
        return self.disc_loss(gt_out, fake_out)
