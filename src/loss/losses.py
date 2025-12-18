import torch 
import torch.nn as nn
import torch.nn.functional as F


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, disc_gt_output, disc_predicted_output):
        loss = 0
        for gt_output, pred_output in zip(disc_gt_output, disc_predicted_output):
            gt_loss = torch.mean((1 - gt_output)**2)
            pred_loss = torch.mean(pred_output**2)
            loss += gt_loss + pred_loss
        return loss

        
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
    def __init__(self):
        super().__init__()
        self.disc_loss = DiscriminatorLoss()
        self.gen_loss = GeneratorLoss()
        self.melspec_loss = MelSpectrogramLoss()
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

        for key in batch:
            if key.endswith("fake_out"):
                disc_name = key[:-9]
                adv_gen_loss = self.gen_loss(batch[key])
                feats_gen_loss = self.fm_loss(batch[f"{disc_name}_gt_feats"], batch[f"{disc_name}_fake_feats"])
                adv_losses[f"{disc_name}_gen_loss"] = adv_gen_loss
                feats_losses[f"{disc_name}_feats_gen_loss"] = feats_gen_loss
                total_loss = total_loss + 2 * adv_gen_loss + feats_gen_loss

        # TODO computation of mel specs here with given melSpecComputer as an argument
        # for better generalization to other spectral losses
        mel_spec_loss = self.melspec_loss(batch["mel_spec_hr"], batch["mel_spec_fake"])
        total_loss = total_loss + 45 * mel_spec_loss
        
        return adv_losses, feats_losses, mel_spec_loss, total_loss
