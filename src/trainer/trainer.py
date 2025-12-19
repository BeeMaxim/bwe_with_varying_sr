from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch.nn.functional as F
from src.metrics.calculate_metrics import calculate_all_metrics
from src.model import HiFiGANWithMRF



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
        if isinstance(self.model, HiFiGANWithMRF):
            wav_fake = self.model.generator(initial_wav, initial_sr, target_sr)
        else:
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
            batch[f"{disc_name}_gt_feats"] = gt_feats
            batch[f"{disc_name}_fake_out"] = fake_out
            batch[f"{disc_name}_fake_feats"] = fake_feats

        batch["mel_spec_fake"] = self.create_mel_spec(batch["generated_wav"].squeeze(1))
        batch["mel_spec_hr"] = self.create_mel_spec(target_wav.squeeze(1))
        
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
            calculate_all_metrics(batch['generated_wav'], batch['wav_hr'], self.metrics["inference"], self.config.datasets.val.initial_sr, self.config.datasets.val.target_sr)

        return batch

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
            self.log_spectrogram(partition='train',**batch)
            self.log_audio(partition='train', **batch)

        else:
            self.log_spectrogram(partition='val', **batch)
            self.log_audio(partition='val',**batch)


    def log_audio(self, wav_lr, wav_hr,  generated_wav, partition, **batch):
        actual_batch_size = min(self.config.dataloader.val.batch_size, len(wav_lr))
        for i in range(actual_batch_size):
            self.writer.add_audio(f"initial_wav_lr_{i}", wav_lr[i][:, :batch['initial_len_lr'][i]], self.config.datasets.val.initial_sr)
            self.writer.add_audio(f"initial_wav_hr_{i}", wav_hr[i][:, :batch['initial_len_hr'][i]], self.config.datasets.val.target_sr)
            self.writer.add_audio(f"generated_wav_{i}", generated_wav[i][:, :batch['initial_len_hr'][i]], self.config.datasets.val.target_sr)


    def log_spectrogram(self, melspec_lr, melspec_hr,  mel_spec_fake, partition, **batch):
        actual_batch_size = min(self.config.dataloader.val.batch_size, len(melspec_lr))
        for i in range(actual_batch_size):
            spectrogram_for_plot_real_lr = melspec_lr[i].detach().cpu()[:, :batch['initial_len_melspec_lr'][i]]
            spectrogram_for_plot_real_hr = melspec_hr[i].detach().cpu()[:, :batch['initial_len_melspec_hr'][i]]
            spectrogram_for_plot_fake = mel_spec_fake[i].detach().cpu()[:, :batch['initial_len_melspec_hr'][i]]
            image = plot_spectrogram(spectrogram_for_plot_real_lr)
            self.writer.add_image(f"melspectrogram_real_lr_{i}", image)
            image_hr = plot_spectrogram(spectrogram_for_plot_real_hr)
            self.writer.add_image(f"melspectrogram_real_hr_{i}", image_hr)
            image_fake = plot_spectrogram(spectrogram_for_plot_fake)
            self.writer.add_image(f"melspectrogram_fake_{i}", image_fake)
        