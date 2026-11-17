"""Generator/critic and data-consistency projection for CelebA SR.

The generator follows the U-Net in Fig. 7 of Ohayon et al. (2023).
The critic follows Table 4 (pre-activation ResNet, f=16 for CelebA).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn
import numpy as np
from librosa.filters import mel as librosa_mel_fn
from torch.nn import functional as F
import math
import torchaudio


mel_basis = {}
hann_window = {}


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool2d(kernel_size=2), DoubleConv(in_channels, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """2x bilinear upsample, concatenate skip + one-channel noise, DoubleConv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = DoubleConv(in_channels + skip_channels + 1, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=True)
        if noise.shape[-2:] != skip.shape[-2:]:
            noise = F.interpolate(noise, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((x, skip, noise), dim=1))


class UNetEstimator(nn.Module):
    """Conditional estimator G(y, z); set stochastic=False for Z_i = 0."""

    def __init__(self, base_channels: int = 24, out_channels: int = 3, in_channels: int = 3) -> None:
        super().__init__()
        s = base_channels
        self.in_channels = in_channels
        self.inc = DoubleConv(in_channels, s)
        self.down1 = DownBlock(s, 2 * s)
        self.down2 = DownBlock(2 * s, 4 * s)
        self.down3 = DownBlock(4 * s, 8 * s)
        self.down4 = DownBlock(8 * s, 8 * s)

        # The channel counts match Fig. 7: (16s+1)->4s, (8s+1)->2s,
        # (4s+1)->s, (2s+1)->s.
        self.up4 = UpBlock(8 * s, 8 * s, 4 * s)
        self.up3 = UpBlock(4 * s, 4 * s, 2 * s)
        self.up2 = UpBlock(2 * s, 2 * s, s)
        self.up1 = UpBlock(s, s, s)
        self.outc = nn.Conv2d(s, out_channels, kernel_size=1)

    @staticmethod
    def _noise(
        batch: int,
        height: int,
        width: int,
        reference: torch.Tensor,
        stochastic: bool,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        if not stochastic:
            return reference.new_zeros((batch, 1, height, width))
        return torch.randn(
            (batch, 1, height, width),
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )

    def forward(
        self,
        low_res_up: torch.Tensor,
        stochastic: bool = False,
        noise_generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        
        if low_res_up.shape[-2] % 16 or low_res_up.shape[-1] % 16:
            raise ValueError("U-Net input height and width must be divisible by 16.")

        x1 = self.inc(low_res_up)       # [B, s, 64, 64]
        x2 = self.down1(x1)             # [B, 2s, 32, 32]
        x3 = self.down2(x2)             # [B, 4s, 16, 16]
        x4 = self.down3(x3)             # [B, 8s, 8, 8]
        x5 = self.down4(x4)             # [B, 8s, 4, 4]

        z4 = self._noise(low_res_up.shape[0], x4.shape[-2], x4.shape[-1], x4,
                         stochastic, noise_generator)
        z3 = self._noise(low_res_up.shape[0], x3.shape[-2], x3.shape[-1], x3,
                         stochastic, noise_generator)
        z2 = self._noise(low_res_up.shape[0], x2.shape[-2], x2.shape[-1], x2,
                         stochastic, noise_generator)
        z1 = self._noise(low_res_up.shape[0], x1.shape[-2], x1.shape[-1], x1,
                         stochastic, noise_generator)

        x = self.up4(x5, x4, z4)
        x = self.up3(x, x3, z3)
        x = self.up2(x, x2, z2)
        x = self.up1(x, x1, z1)
        # The paper's Fig. 7 has no activation after this final convolution.
        return self.outc(x)


class PreActResBlock(nn.Module):
    """Pre-activation ResNet block with 0.1 residual scaling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.activation = nn.LeakyReLU(0.2, inplace=False)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv1(self.activation(x))
        residual = self.conv2(self.activation(residual))
        return self.skip(x) + 0.1 * residual



def block_average_downsample(image: torch.Tensor, scale: int) -> torch.Tensor:
    """Plain non-overlapping block averaging, as used in the paper."""
    if image.shape[-2] % scale or image.shape[-1] % scale:
        raise ValueError(f"Image dimensions must be divisible by scale={scale}.")
    return F.avg_pool2d(image, kernel_size=scale, stride=scale)


def nearest_upsample(low_res: torch.Tensor, out_size: tuple[int, int] = (64, 64)) -> torch.Tensor:
    return F.interpolate(low_res, size=out_size, mode="nearest")


def enforce_consistency(
    prediction: torch.Tensor,
    low_res: torch.Tensor,
    scale: int,
) -> torch.Tensor:
    """Orthogonal projection onto {x: block_average_downsample(x) == low_res}.

    For non-overlapping average pooling, H^T(HH^T)^-1 is exactly replication
    of each LR residual over its corresponding scale x scale HR block.
    This is the linear CEM projection; it is not a pixel reconstruction loss.
    """
    current_lr = block_average_downsample(prediction, scale)
    residual = low_res - current_lr
    correction = F.interpolate(residual, scale_factor=scale, mode="nearest")
    return prediction + correction



def stft(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=True):

    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y)

    if y.dim() == 1:
        y = y.unsqueeze(0)

    window = torch.hann_window(
        win_size, device=y.device, dtype=y.dtype
    )

    spec = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )

    return torch.abs(spec), spec


def istft(spec, n_fft, hop_size, win_size,
          length=None, center=True):

    window = torch.hann_window(
        win_size, device=spec.device, dtype=spec.real.dtype
    )

    return torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window,
        center=center,
        normalized=False,
        onesided=True,
        length=length,
    )


class SuperResolutionGenerator(nn.Module):
    def __init__(
        self,
        base_channels = 24,
        stochastic = False,
        in_channels = 2,     
        n_fft = 256,
        hop_size = 64,
        win_size = 256
    ):
        super().__init__()
        self.stochastic = stochastic
        self.estimator = UNetEstimator(
            base_channels=base_channels,
            in_channels=in_channels,
            out_channels=in_channels,
        )
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size

        self.input_conv = nn.Conv1d(129 * 2, 256, 1)
        self.output_conv = nn.Conv1d(256, 129 * 2, 1)

    def get_stft(self, x, sampling_rate):
        shape = x.shape
        x = x.view(shape[0] * shape[1], shape[2])
        x = stft(x, n_fft=self.n_fft, num_mels=80, sampling_rate=sampling_rate, hop_size=self.hop_size, win_size=self.win_size, fmin=None, fmax=None)[1]
        x = x.view(shape[0], -1, x.shape[-1])
        return x

    def get_stft(self, x, sampling_rate):
        shape = x.shape

        x = x.reshape(shape[0] * shape[1], shape[2])

        x = stft(
            x,
            n_fft=self.n_fft,
            num_mels=80,
            sampling_rate=sampling_rate,
            hop_size=self.hop_size,
            win_size=self.win_size,
            fmin=None,
            fmax=None,
        )[1]

        x = x.reshape(shape[0], -1, x.shape[-1])

        return x


    def get_istft(self, x, original_length):
        batch, _, frames = x.shape

        freq_bins = self.n_fft // 2 + 1
        channels = x.shape[1] // freq_bins

        x = x.reshape(batch * channels, freq_bins, frames)

        x = istft(
            x,
            n_fft=self.n_fft,
            hop_size=self.hop_size,
            win_size=self.win_size,
            length=original_length,
        )

        return x.reshape(batch, channels, -1)

    def forward(self, x_lr, initial_sr, target_sr, **batch):
        x = x_lr.clone()

        closest_size = ((x.shape[-1] + self.hop_size - 1) // self.hop_size) * self.hop_size
        pad_size =  closest_size - x.shape[-1]
        x = torch.nn.functional.pad(x, (0, pad_size))
        x = torchaudio.functional.resample(x, initial_sr, target_sr)
        resampled_length = x.shape[-1]
        x = self.get_stft(x, sampling_rate=target_sr)
        x = torch.stack([x.real, x.imag], dim=1)

        padding = (int(math.ceil(x.shape[-1] / 16.0)) * 16 - x.shape[-1])
        padding_right = padding // 2
        padding_left = padding - padding_right

        x = torch.nn.functional.pad(x, (padding_left, padding_right))

        x = self.input_conv(x.reshape(x.shape[0], 129 * 2, -1)).reshape(x.shape[0], 2, 128, -1)

        x = self.estimator(x, stochastic=self.stochastic)

        x = self.output_conv(x.reshape(x.shape[0], 2 * 128, -1)).reshape(x.shape[0], 2, 129, -1)

        spec = torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous())
        x = self.get_istft(spec, original_length=resampled_length)

        return x[..., :x_lr.shape[-1] * (target_sr // initial_sr)]


class StochGAN(nn.Module):
    def __init__(self,
                 generator_config,
                 discriminators):
        super().__init__()

        self.discriminators = nn.ModuleDict()
    
        for disc_name, disc in discriminators.items():
            self.discriminators[disc_name] = disc

        self.generator = SuperResolutionGenerator(**generator_config)


    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        gen_parameters = sum(
            [p.numel() for p in self.generator.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = ""
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"
        result_info = result_info + f"\nGen: {gen_parameters}"
        for disc_name, disc in self.discriminators.items():
            parameters = sum(
                [p.numel() for p in disc.parameters() if p.requires_grad]
            )
            result_info = result_info + f"\n{disc_name}: {parameters}"

        return result_info
