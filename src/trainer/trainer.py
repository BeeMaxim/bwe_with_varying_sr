from pathlib import Path

import pandas as pd
import torch

from src.logger.utils import plot_spectrogram, plot_std_map, plot_freq_std
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

        candidates = []

        if self.is_train:
            self.gen_optimizer.zero_grad()  

        K = self.exploration
        generator_loss = 0
        best_rec_loss = None
        mean_rec_loss = 0

        for disc_name in self.model.discriminators.keys():    
            batch[f"{disc_name}_feats_loss"] = 0
            batch[f"{disc_name}_gen_adv_loss"] = 0

        for _ in range(K):
            wav_fake = self.model.generator(initial_wav, initial_sr, target_sr, **batch)
            if target_wav.shape != wav_fake.shape:
                wav_fake = torch.stack([F.pad(wav, (0, target_wav.shape[2] - wav_fake.shape[2]), value=0) for wav in wav_fake])
                  
            total_gan_loss = 0

            if self.use_adversarial:
                for disc_name, disc in self.model.discriminators.items():
                    _, gt_feats, fake_out, fake_feats = disc(target_wav, wav_fake, **batch)
                    adv_loss, feats_loss, gan_loss = self.criterion.exploration_adversarial_loss(gt_feats, fake_out, fake_feats)
                    total_gan_loss += gan_loss / K

                    batch[f"{disc_name}_feats_loss"] += feats_loss.detach().item() / K
                    batch[f"{disc_name}_gen_adv_loss"] += adv_loss.detach().item() / K
                    generator_loss += gan_loss.detach().item() / K

                if self.is_train:
                    total_gan_loss.backward(retain_graph=True)

            B = initial_wav.shape[0]
            if best_rec_loss is None:
                best_rec_loss = torch.ones(B) * -1
            for b in range(B):
                cur_rec_loss = self.criterion.exploration_reconstruction_loss(wav_fake[b:b+1], target_wav[b:b+1])
                mean_rec_loss += cur_rec_loss.detach().item()
                if best_rec_loss[b] < 0 or cur_rec_loss < best_rec_loss[b]:
                    best_rec_loss[b] = cur_rec_loss

            candidates.append(wav_fake.detach())

        generator_loss += best_rec_loss.mean().detach().item()
        batch["best_reconstruction_loss"] = best_rec_loss.mean().detach().item()
        batch["mean_reconstruction_loss"] = mean_rec_loss / (B * K)
        batch["generator_loss"] = generator_loss

        if self.is_train:
            best_rec_loss.mean().backward()
            self._clip_grad_norm(self.model.generator)
            self.gen_optimizer.step()
            if self.use_adversarial:
                self.disc_optimizer.zero_grad()

        for disc_name in self.model.discriminators.keys():      
            batch[f"{disc_name}_disc_adv_loss"] = 0
    
        if self.use_adversarial:
            for cand in candidates:
                disc_loss = 0
                for disc_name, disc in self.model.discriminators.items():
                    gt_out, _, fake_out, _ = disc(target_wav, cand)
                    gt_loss, fake_loss = self.criterion.exploration_discriminator_loss(gt_out, fake_out)
                    disc_loss += gt_loss / K + fake_loss / K

                    batch[f"{disc_name}_disc_adv_loss"] += gt_loss.detach().item() / K + fake_loss.detach().item() / K

                if self.is_train:
                    disc_loss.backward()

            if self.is_train:
                for _, disc in self.model.discriminators.items():
                    self._clip_grad_norm(disc)    
                self.disc_optimizer.step()

        batch["spec_gt"] = self.create_spec(target_wav.squeeze(1))
        batch["spec_fake"] = self.create_spec(candidates[0].squeeze(1))
        batch["generated_wav"] = candidates[0]

        if not self.is_train:
            S = self.std_estimation
            if S < K:
                samples = candidates[:S]
            else:
                samples = candidates.copy() # mix of exploration samples and new ones (only in val now)
                for _ in range(S - K):
                    wav_fake = self.model.generator(initial_wav, initial_sr, target_sr, **batch)
                    if target_wav.shape != wav_fake.shape:
                        wav_fake = torch.stack([F.pad(wav, (0, target_wav.shape[2] - wav_fake.shape[2]), value=0) for wav in wav_fake])
                    samples.append(wav_fake)

            batch["samples"] = samples

        for loss_name in batch.keys():
            if not loss_name.endswith("loss"):
                continue
            metrics.update(loss_name, batch[loss_name])

        if not self.is_train:
            _, std_maps = calculate_all_metrics(batch['generated_wav'], batch['wav_hr'], self.metrics["inference"], self.config.datasets.val.initial_sr, self.config.datasets.val.target_sr, **batch)
            batch["std_maps"] = std_maps

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
            self.log_spectrogram(batch_idx, partition='train', **batch)
            self.log_audio(batch_idx, partition='train', **batch)

        else:
            self.log_spectrogram(batch_idx, partition='val', **batch)
            self.log_audio(batch_idx, partition='val', **batch)
            self.log_std_maps(batch_idx, partition='val', **batch)


    def log_audio(self, batch_idx, wav_lr, wav_hr, generated_wav, partition, **batch):
        actual_batch_size = min(self.config.dataloader.val.batch_size, len(wav_lr))
        for i in range(actual_batch_size):
            self.writer.add_audio(f"initial_wav_lr_{batch_idx}_{i}", wav_lr[i][:, :batch['initial_len_lr'][i]], self.config.datasets.val.initial_sr)
            self.writer.add_audio(f"initial_wav_hr_{batch_idx}_{i}", wav_hr[i][:, :batch['initial_len_hr'][i]], self.config.datasets.val.target_sr)
            self.writer.add_audio(f"generated_wav_{batch_idx}_{i}", generated_wav[i][:, :batch['initial_len_hr'][i]], self.config.datasets.val.target_sr)
            

    def log_spectrogram(self, batch_idx, partition, **batch):
        actual_batch_size = min(self.config.dataloader.val.batch_size, len(batch["spec_gt"]))
        for i in range(actual_batch_size):
            image_spec = plot_spectrogram(batch["spec_gt"][i].detach().cpu())
            self.writer.add_image(f"{batch_idx}_{i}_spectrogram_real_hr", image_spec)
            
            image_spec = plot_spectrogram(batch["spec_fake"][i].detach().cpu())
            self.writer.add_image(f"{batch_idx}_{i}_spectrogram_fake_hr", image_spec)

    def log_std_maps(self, batch_idx, partition, **batch):
        actual_batch_size = min(self.config.dataloader.val.batch_size, len(batch["spec_gt"]))
        for i in range(actual_batch_size):
            std_map = plot_std_map(batch["std_maps"].detach().cpu())
            self.writer.add_image(f"{batch_idx}_{i}_std_map", std_map)
            std_freq = plot_freq_std(batch["std_maps"].mean(-1).detach().cpu())
            self.writer.add_image(f"{batch_idx}_{i}_std_freq", std_freq)
        