from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch.nn.functional as F
from src.metrics.calculate_metrics import calculate_all_metrics


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """
    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)

        initial_wav = batch['wav_lr']
        target_wav = batch['wav_hr']
        initial_sr = self.config.datasets.train.initial_sr
        target_sr = self.config.datasets.train.target_sr

        wav_fake, outs = self.model.generator(initial_wav, initial_sr, target_sr, **batch)
        
        if target_wav.shape != wav_fake.shape:
            wav_fake = torch.stack([F.pad(wav, (0, target_wav.shape[2] - wav_fake.shape[2]), value=0) for wav in wav_fake])
        
        batch["generated_wav"] = wav_fake
        batch["generator_outs"] = outs

        if self.is_train:
            self.disc_optimizer.zero_grad()

        for disc_name, disc in self.model.discriminators.items():
            detached_outs = [x.detach() for x in outs]
            gt_out, _, fake_out, _ = disc(target_wav, wav_fake.detach(), generator_outs=detached_outs)
            batch[f"{disc_name}_gt_out"] = gt_out
            batch[f"{disc_name}_fake_out"] = fake_out

        disc_losses, disc_loss = self.criterion.discriminator_loss(batch)

        if self.is_train:
            disc_loss.backward()
            for _, disc in self.model.discriminators.items():
                self._clip_grad_norm(disc)     
            self.disc_optimizer.step()
            self.gen_optimizer.zero_grad()

        # disc call for generator loss
        for disc_name, disc in self.model.discriminators.items():
            _, gt_feats, fake_out, fake_feats = disc(target_wav, wav_fake, **batch)
            if not self.is_train:
                disc.train()
                batch[f"{disc_name}_step_1"] = self._get_audio_from_disc_grad(disc, step=1, **batch)
                batch[f"{disc_name}_step_1_spec"] = self.create_spec(batch[f"{disc_name}_step_1"].squeeze(1))
                batch[f"{disc_name}_step_5"] = self._get_audio_from_disc_grad(disc, step=5, **batch)
                batch[f"{disc_name}_step_5_spec"] = self.create_spec(batch[f"{disc_name}_step_5"].squeeze(1))
                disc.eval()

            batch[f"{disc_name}_gt_feats"] = gt_feats
            batch[f"{disc_name}_fake_out"] = fake_out
            batch[f"{disc_name}_fake_feats"] = fake_feats

        batch["mel_spec_fake"] = self.create_mel_spec(batch["generated_wav"].squeeze(1))
        batch["mel_spec_hr"] = self.create_mel_spec(target_wav.squeeze(1))
        batch["spec_hr"] = self.create_spec(target_wav.squeeze(1))
        batch["spec_fake"] = self.create_spec(wav_fake.squeeze(1))
        
        adv_gen_losses, feats_gen_losses, mel_spec_loss, gen_loss = self.criterion.generator_loss(batch)

        if self.is_train:
            gen_loss.backward()
            self._clip_grad_norm(self.model.generator)      
            self.gen_optimizer.step()

        batch["disc_loss"] = disc_loss

        batch.update(adv_gen_losses)
        batch.update(feats_gen_losses)
        batch.update(disc_losses)
        batch["mel_spec_loss"] = mel_spec_loss
        batch["gen_loss"] = gen_loss

        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        if not self.is_train:
            disc_name = list(self.model.discriminators.keys())[0]
            calculate_all_metrics(batch['generated_wav'], batch['wav_hr'], self.metrics["inference"], self.config.datasets.val.initial_sr, self.config.datasets.val.target_sr)
            calculate_all_metrics(batch[f"{disc_name}_step_1"], batch['wav_hr'], self.metrics["inference_step1"], self.config.datasets.val.initial_sr, self.config.datasets.val.target_sr)
            calculate_all_metrics(batch[f"{disc_name}_step_5"], batch['wav_hr'], self.metrics["inference_step5"], self.config.datasets.val.initial_sr, self.config.datasets.val.target_sr)

        return batch

    def _get_audio_from_disc_grad(self, disc, step=1, **batch):
        x = batch["generated_wav"]

        for _ in range(step):
            x = self._get_audio_from_disc_grad_step(disc, x, **batch)

        return x

    def _get_audio_from_disc_grad_step(self, disc, current_outs, **batch):
        current_outs.requires_grad_(True)

        with torch.enable_grad():
            _, _, dsc_output, _ = disc(current_outs, current_outs, **batch)
            loss = 0.0
            for predicted in dsc_output:
                pred_loss = torch.mean((1 - predicted)**2)
                loss += pred_loss
            grad = torch.autograd.grad(loss, current_outs)[0]

        # grad = grad / (grad.flatten(1).norm(dim=1, keepdim=True) + 1e-8)
        eta = 3e-4
        x = (current_outs - eta * grad).detach()

        return x

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        if mode == "train": 
            self.log_spectrogram(batch_idx, partition='train', **batch)
            self.log_audio(batch_idx, partition='train', **batch)

        else:
            self.log_spectrogram(batch_idx, partition='val', **batch)
            self.log_audio(batch_idx, partition='val', **batch)


    def log_audio(self, batch_idx, wav_lr, wav_hr, generated_wav, partition, **batch):
        actual_batch_size = min(self.config.dataloader.val.batch_size, len(wav_lr))
        for i in range(actual_batch_size):
            self.writer.add_audio(f"initial_wav_lr_{batch_idx}_{i}", wav_lr[i][:, :batch['initial_len_lr'][i]], self.config.datasets.val.initial_sr)
            self.writer.add_audio(f"initial_wav_hr_{batch_idx}_{i}", wav_hr[i][:, :batch['initial_len_hr'][i]], self.config.datasets.val.target_sr)
            self.writer.add_audio(f"generated_wav_{batch_idx}_{i}", generated_wav[i][:, :batch['initial_len_hr'][i]], self.config.datasets.val.target_sr)
            for key, value in batch.items():
                if key.endswith("step_1"):
                    disc = key.split('_')[0]
                    self.writer.add_audio(f"{disc}_shift_{batch_idx}_{i}_step_1", value[i][:, :batch['initial_len_hr'][i]], self.config.datasets.val.target_sr)
                if key.endswith("step_5"):
                    disc = key.split('_')[0]
                    self.writer.add_audio(f"{disc}_shift_{batch_idx}_{i}_step_5", value[i][:, :batch['initial_len_hr'][i]], self.config.datasets.val.target_sr)


    def log_spectrogram(self, batch_idx, melspec_lr, melspec_hr, mel_spec_fake, partition, **batch):
        actual_batch_size = min(self.config.dataloader.val.batch_size, len(melspec_lr))
        for i in range(actual_batch_size):
            for key, value in batch.items():
                if key.endswith("step_1_spec"):
                    disc = key.split('_')[0]
                    image = plot_spectrogram(value[i].detach().cpu())
                    self.writer.add_image(f"{batch_idx}_{i}_{disc}_shift_step_1_spec", image)
                if key.endswith("step_5_spec"):
                    disc = key.split('_')[0]
                    image = plot_spectrogram(value[i].detach().cpu())
                    self.writer.add_image(f"{batch_idx}_{i}_{disc}_shift_step_5_spec", image)

            spectrogram_for_plot_real_lr = melspec_lr[i].detach().cpu()[:, :batch['initial_len_melspec_lr'][i]]
            spectrogram_for_plot_real_hr = melspec_hr[i].detach().cpu()[:, :batch['initial_len_melspec_hr'][i]]
            spectrogram_for_plot_fake = mel_spec_fake[i].detach().cpu()[:, :batch['initial_len_melspec_hr'][i]]
            image = plot_spectrogram(spectrogram_for_plot_real_lr)
            self.writer.add_image(f"melspectrogram_real_lr_{batch_idx}_{i}", image)
            image_hr = plot_spectrogram(spectrogram_for_plot_real_hr)
            self.writer.add_image(f"melspectrogram_real_hr_{batch_idx}_{i}", image_hr)
            image_fake = plot_spectrogram(spectrogram_for_plot_fake)
            self.writer.add_image(f"melspectrogram_fake_{batch_idx}_{i}", image_fake)

            image_spec = plot_spectrogram(batch["spec_hr"][i].detach().cpu())
            self.writer.add_image(f"{batch_idx}_{i}_spectrogram_real_hr", image_spec)
            
            image_spec = plot_spectrogram(batch["spec_fake"][i].detach().cpu())
            self.writer.add_image(f"{batch_idx}_{i}_spectrogram_fake_hr", image_spec)
        