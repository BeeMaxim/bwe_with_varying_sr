from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import  spectral_norm #, weight_norm
from torch.nn.utils.parametrizations import weight_norm
from librosa.filters import mel as librosa_mel_fn
import numpy as np
import src.utils.upsampling_utils as upsampling_utils
import librosa
from librosa.util import normalize
from src.model.melspec import MelSpectrogram
from src.model.wavehax_generator import WavehaxGenerator
import torchaudio
from src.model.baenet import Generator
import pyworld
from pesq import pesq
from pystoi import stoi
from src.model.bigvgan import BigVUpsampling
from src.model.rndvoc import RNDVocoder16k
from src.model.comvo import ComVo, ConvNeXtBlock
from src.model.generator import HiFiPlusGenerator, stft, STFTMag
from torch.utils.checkpoint import checkpoint

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#plt.switch_backend("agg")

from pathlib import Path


def closest_power_of_two(n):
    return 1 << (n - 1).bit_length()


class LN(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


import math

import torch
from torch import nn
from torch.nn import functional as F

'''
class SpectralConv2d(nn.Module):
    """Fourier convolution for [B, C, F, T]."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_f: int = 8,
        modes_t: int = 16,
    ):
        super().__init__()

        if min(in_channels, out_channels, modes_f, modes_t) < 1:
            raise ValueError("Channel counts and modes must be positive.")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_f = modes_f
        self.modes_t = modes_t

        scale = 1.0 / math.sqrt(in_channels * out_channels)
        shape = (
            in_channels, out_channels, modes_f, modes_t, 2
        )

        # Last dimension stores real and imaginary components.
        self.weight_pos = nn.Parameter(
            scale * torch.randn(*shape)
        )
        self.weight_neg = nn.Parameter(
            scale * torch.randn(*shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected [B,C,F,T], got {tuple(x.shape)}"
            )

        input_dtype = x.dtype
        x32 = x.float()

        b, _, freq, time = x32.shape

        x_ft = torch.fft.rfft2(
            x32, dim=(-2, -1), norm="ortho"
        )

        time_freq = x_ft.shape[-1]

        out_ft = torch.zeros(
            b,
            self.out_channels,
            freq,
            time_freq,
            dtype=x_ft.dtype,
            device=x.device,
        )

        # Limit modes for variable-sized inputs.
        mf = min(self.modes_f, freq // 2)
        mt = min(self.modes_t, time_freq)

        if mf > 0 and mt > 0:
            w_pos = torch.view_as_complex(
                self.weight_pos.float().contiguous()
            )
            w_neg = torch.view_as_complex(
                self.weight_neg.float().contiguous()
            )

            out_ft[:, :, :mf, :mt] = torch.einsum(
                "bift,ioft->boft",
                x_ft[:, :, :mf, :mt],
                w_pos[:, :, :mf, :mt],
            )

            out_ft[:, :, -mf:, :mt] = torch.einsum(
                "bift,ioft->boft",
                x_ft[:, :, -mf:, :mt],
                w_neg[:, :, :mf, :mt],
            )

        y = torch.fft.irfft2(
            out_ft,
            s=(freq, time),
            dim=(-2, -1),
            norm="ortho",
        )

        return y.to(dtype=input_dtype)


class LocalResidualBlock(nn.Module):
    """Local time-frequency mixing."""

    def __init__(
        self,
        channels: int,
        groups: int = 8,
        kernel_size: tuple[int, int] = (5, 7),
    ):
        super().__init__()

        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")

        kf, kt = kernel_size

        self.norm = nn.GroupNorm(groups, channels)

        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=(kf // 2, kt // 2),
            groups=channels,
        )

        self.expand = nn.Conv2d(
            channels, 2 * channels, kernel_size=1
        )
        self.contract = nn.Conv2d(
            2 * channels, channels, kernel_size=1
        )

        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.depthwise(h)
        h = self.contract(F.gelu(self.expand(h)))
        return x + self.scale * h


class FourierResidualBlock(nn.Module):
    """Low-rank global Fourier mixing + local channel mixing."""

    def __init__(
        self,
        channels: int = 64,
        spectral_channels: int = 16,
        modes_f: int = 8,
        modes_t: int = 16,
        groups: int = 8,
    ):
        super().__init__()

        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")

        self.norm = nn.GroupNorm(groups, channels)

        self.reduce = nn.Conv2d(
            channels, spectral_channels, kernel_size=1
        )

        self.spectral = SpectralConv2d(
            spectral_channels,
            spectral_channels,
            modes_f=modes_f,
            modes_t=modes_t,
        )

        # Local channel mixing complements the global Fourier path.
        self.local = nn.Conv2d(
            spectral_channels, spectral_channels, kernel_size=1
        )

        self.expand = nn.Conv2d(
            spectral_channels, 2 * spectral_channels, kernel_size=1
        )
        self.contract = nn.Conv2d(
            2 * spectral_channels, spectral_channels, kernel_size=1
        )
        self.project = nn.Conv2d(
            spectral_channels, channels, kernel_size=1
        )

        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.reduce(self.norm(x))

        h = self.spectral(h) + self.local(h)
        h = F.gelu(h)
        h = self.contract(F.gelu(self.expand(h)))

        return x + self.scale * self.project(h)


class SpectrogramBottleneck(nn.Module):
    """
    Input:  [B, 129, T]
    Output: [B, 129, T]

    T can vary between calls.
    """

    def __init__(
        self,
        freq_bins: int = 129,
        channels: int = 64,
        local_blocks: int = 4,
        fourier_blocks: int = 2,
        spectral_channels: int = 16,
        modes_f: int = 8,
        modes_t: int = 16,
        groups: int = 8,
    ):
        super().__init__()

        if freq_bins < 1:
            raise ValueError("freq_bins must be positive")
        if channels % groups != 0:
            raise ValueError("channels must be divisible by groups")
        if local_blocks < 0 or fourier_blocks < 0:
            raise ValueError("Block counts must be nonnegative")

        self.freq_bins = freq_bins

        self.input_proj = nn.Conv2d(
            1, channels, kernel_size=1
        )

        blocks = []

        # Interleave local and global processing.
        for i in range(max(local_blocks, fourier_blocks)):
            if i < local_blocks:
                blocks.append(
                    LocalResidualBlock(channels, groups=groups)
                )

            if i < fourier_blocks:
                blocks.append(
                    FourierResidualBlock(
                        channels=channels,
                        spectral_channels=spectral_channels,
                        modes_f=modes_f,
                        modes_t=modes_t,
                        groups=groups,
                    )
                )

        self.blocks = nn.Sequential(*blocks)

        self.output_norm = nn.GroupNorm(groups, channels)
        self.output_proj = nn.Conv2d(
            channels, 1, kernel_size=1
        )
        self.output_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected [B,F,T], got {tuple(x.shape)}"
            )

        if x.shape[1] != self.freq_bins:
            raise ValueError(
                f"Expected F={self.freq_bins}, got F={x.shape[1]}"
            )

        if not x.is_floating_point():
            raise TypeError("Input must be a real floating-point tensor")

        if x.shape[-1] < 1:
            raise ValueError("Time dimension T must be >= 1")

        x4 = x.unsqueeze(1)  # [B, 1, F, T]

        h = self.input_proj(x4)
        h = self.blocks(h)
        h = F.gelu(self.output_norm(h))

        delta = self.output_proj(h)

        # Residual output: preserves the input shape and information.
        y = x4 + self.output_scale * delta

        return y.squeeze(1)  # [B, F, T]'''

import math
import torch
from torch import nn
from torch.nn import functional as F


class FourierBlock(nn.Module):
    def __init__(self, c=32, inner=8, modes=16):
        super().__init__()
        self.norm = nn.GroupNorm(4, c)
        self.reduce = nn.Conv2d(c, inner, 1)
        self.weight = nn.Parameter(
            torch.randn(inner, inner, modes, dtype=torch.cfloat)
            / math.sqrt(inner)
        )
        self.local = nn.Conv2d(inner, inner, 1)
        self.ff = nn.Sequential(
            nn.Conv2d(inner, 2 * inner, 1),
            nn.GELU(),
            nn.Conv2d(2 * inner, inner, 1),
        )
        self.project = nn.Conv2d(inner, c, 1)
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.modes = modes

    def forward(self, x):
        h = self.reduce(self.norm(x))
        t = h.shape[-1]

        hf = torch.fft.rfft(h.float(), dim=-1, norm="ortho")
        m = min(self.modes, hf.shape[-1])

        yf = torch.zeros_like(hf)
        yf[..., :m] = torch.einsum(
            "bift,iot->boft", hf[..., :m], self.weight[..., :m]
        )
        h = torch.fft.irfft(yf, n=t, dim=-1, norm="ortho").to(x.dtype)

        h = F.gelu(h + self.local(self.reduce(self.norm(x))))
        return x + self.scale * self.project(self.ff(h))


class LocalBlock(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(4, c),
            nn.Conv2d(c, c, (3, 5), padding=(1, 2), groups=c),
            nn.Conv2d(c, 2 * c, 1),
            nn.GELU(),
            nn.Conv2d(2 * c, c, 1),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        return x + self.scale * self.net(x)


class SpectrogramBottleneck(nn.Module):
    def __init__(self, channels=32, modes=16):
        super().__init__()
        self.input = nn.Conv2d(1, channels, 1)
        self.blocks = nn.Sequential(
            LocalBlock(channels),
            FourierBlock(channels, inner=8, modes=modes),
            LocalBlock(channels),
        )
        self.output = nn.Sequential(
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # x: [B, 129, T]
        x = x.unsqueeze(1)
        h = self.blocks(self.input(x))
        return (x + self.scale * self.output(h)).squeeze(1)
    

class ExploreAE(HiFiPlusGenerator):
    def __init__(
        self,
        hifi_resblock="1",
        hifi_upsample_rates=(8, 8, 2, 2),
        hifi_upsample_initial_channel=128,
        hifi_resblock_kernel_sizes=(3, 7, 11),
        hifi_resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        hifi_conv_pre_kernel_size=1,

        upsample_init_channels = 1,
        upsample_block_rates=[2, 2],
        upsample_block_kernel_sizes=[4, 4], 

        residual_channels=64,
        bsft_channels=64,
        nwstack1_blocks=1,
        nwstack2_blocks=1,

        use_spectralunet=True,
        spectralunet_block_widths=(8, 16, 24, 32, 64),
        spectralunet_block_depth=5,
        spectralunet_positional_encoding=True,

        use_waveunet=True,
        waveunet_block_widths=(10, 20, 40, 80),
        waveunet_block_depth=4,

        use_spectralmasknet=True,

        norm_type: Literal["weight", "spectral"] = "weight",
        use_skip_connect=True,

        waveunet_input: Literal["waveform", "hifi", "both"] = "both",

        activation: Literal["leaky_relu", "snake"] = "leaky_relu",
   
        n_fft=1024,
        hop_size=256,
        win_size=1024
    ):
        hifi_input_channels = n_fft // 2 + 1

        super().__init__(
            hifi_resblock=hifi_resblock,
            hifi_upsample_rates=hifi_upsample_rates,
            hifi_upsample_initial_channel=hifi_upsample_initial_channel,
            hifi_resblock_kernel_sizes=hifi_resblock_kernel_sizes,
            hifi_resblock_dilation_sizes=hifi_resblock_dilation_sizes,
            hifi_input_channels=hifi_input_channels,
            hifi_conv_pre_kernel_size=hifi_conv_pre_kernel_size,

            upsample_init_channels=upsample_init_channels,
            upsample_block_rates=upsample_block_rates,
            upsample_block_kernel_sizes=upsample_block_kernel_sizes,

            residual_channels=residual_channels,
            bsft_channels=bsft_channels,
            nwstack1_blocks=nwstack1_blocks,
            nwstack2_blocks=nwstack2_blocks,

            use_spectralunet=use_spectralunet,
            spectralunet_block_widths=spectralunet_block_widths,
            spectralunet_block_depth=spectralunet_block_depth,
            spectralunet_positional_encoding=spectralunet_positional_encoding,
            kernel_sizes_mrf=None,
            dilations_mrf=None,

            use_waveunet=use_waveunet,
            waveunet_block_widths=waveunet_block_widths,
            waveunet_block_depth=waveunet_block_depth,

            use_spectralmasknet=use_spectralmasknet,

            norm_type=norm_type,
            use_skip_connect=use_skip_connect,

            activation=activation
        )

        self.waveunet_input = waveunet_input

        self.waveunet_conv_pre = None 
        if self.waveunet_input == "waveform":
            self.waveunet_conv_pre = weight_norm(
                nn.Conv1d(
                    1, self.hifi.out_channels, 1
                )
            )
        elif self.waveunet_input == "both":
            self.waveunet_conv_pre = weight_norm(
                nn.Conv1d(
                    1 + self.hifi.out_channels, self.hifi.out_channels, 1
                )
            )

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 4, 8, stride=4, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(4, 32, 8, stride=4, padding=2),
            nn.LeakyReLU(),
            nn.Conv1d(32, 129, 8, stride=4, padding=2),
        )
        
        self.bottleneck = nn.Sequential(
            # nn.LeakyReLU(),
            nn.Conv1d(129, 64, 1, 1),
            nn.LeakyReLU(),
            nn.Conv1d(64, 129, 1, 1)
        )

        self.bottleneck = SpectrogramBottleneck()

        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size


    def get_stft(self, x, sampling_rate):
        shape = x.shape
        x = x.view(shape[0] * shape[1], shape[2])
        x = stft(x, n_fft=self.n_fft, num_mels=80, sampling_rate=sampling_rate, hop_size=self.hop_size, win_size=self.win_size, fmin=None, fmax=None)[1]
        x = x.view(shape[0], -1, x.shape[-1])
        return x
    
    
    def apply_waveunet_a2a(self, x, x_reference):
        if self.waveunet_input == "waveform":
            x_a = self.waveunet_conv_pre(x_reference)
        elif self.waveunet_input == "both":
            x_a = torch.cat([x, x_reference], 1)
            x_a = self.waveunet_conv_pre(x_a)
        elif self.waveunet_input == "hifi":
            x_a = x
        else:
            raise ValueError
        x = self.waveunet(x_a)
        if self.use_skip_connect:
            x += self.waveunet_skip_connect(x_a)
        return x

    def forward(self, x, initial_sr, target_sr, latent=None, **batch):
        initial_x = x.clone()
        batch_size = x.shape[0]

        current_size = initial_x.shape[-1]
        target_size = (target_sr // initial_sr) * current_size
        closest_size = ((current_size + 1023) // 1024) * 1024
        pad_size =  closest_size - current_size
        padded_x = torch.nn.functional.pad(initial_x, (0, pad_size))
        expected_reference_len = (closest_size * target_sr) // initial_sr


        if batch['mode'] == 'train':
            x_reference = batch['reference_wav']
        else:
            resampled_audio = []
            for i in range(batch_size):
                x_single = padded_x[i].cpu().numpy()

                x_resampled = librosa.resample(
                    x_single, orig_sr=initial_sr, target_sr=target_sr, res_type="polyphase"
                )

                target_length_resempled = x_single.shape[-1] * (target_sr // 2 // initial_sr)
                if len(x_resampled) > target_length_resempled:
                    x_resampled = x_resampled[:target_length_resempled]
                
                resampled_audio.append(x_resampled)
            x_reference = np.stack(resampled_audio)
            x_reference = torch.tensor(x_reference, dtype=padded_x.dtype).to(x.device)

        current_reference_len = x_reference.shape[-1]
        pad_reference_len = expected_reference_len - current_reference_len

        padded_reference = torch.nn.functional.pad(x_reference, (0, pad_reference_len)).to(x.device)
        
        x_res = padded_reference

        x_res = self.encoder(x_res)
        lf = 129 // 4
        if latent is None:
            x_res[:, :lf] += torch.randn_like(x_res[:, :lf]).to(x_res.device) / 3
        else:
            best_latent = None
            best_mse = None
            for i in range(3):
                cur_latent = x_res.clone()
                cur_latent[:, :lf] += torch.randn_like(x_res[:, :lf]).to(x_res.device) / 3
                with torch.no_grad():
                    est_latent = self.bottleneck(cur_latent)
                mse = F.mse_loss(est_latent, latent, reduction="none").mean((1, 2))
                if best_latent is None:
                    best_latent = cur_latent
                    best_mse = mse
                else:
                    for b in range(mse.size(0)):
                        if mse[b] < best_mse[b]:
                            best_mse[b] = mse[b]
                            best_latent[b] = latent[b]
            x_res = best_latent

        x_res = self.bottleneck(x_res)

        x_res, _ = self.hifi(x_res)
    
        if self.use_waveunet:
            x_res = self.apply_waveunet_a2a(x_res, padded_reference)

        if self.use_spectralmasknet:
            x_res = self.apply_spectralmasknet(x_res)

        x_res = self.conv_post(x_res)
        x_res = torch.tanh(x_res)

        return x_res[..., :target_size]


    def forward_hr(self, x_lr, x_hr, initial_sr, target_sr, **batch):
        initial_x = x_lr.clone()
        batch_size = x_lr.shape[0]

        current_size = initial_x.shape[-1]
        target_size = (target_sr // initial_sr) * current_size
        closest_size = ((current_size + 1023) // 1024) * 1024
        pad_size =  closest_size - current_size
        padded_x = torch.nn.functional.pad(initial_x, (0, pad_size))
        expected_reference_len = (closest_size * target_sr) // initial_sr

        if batch['mode'] == 'train':
            x_reference = batch['reference_wav']
        else:
            resampled_audio = []
            for i in range(batch_size):
                x_single = padded_x[i].cpu().numpy()

                x_resampled = librosa.resample(
                    x_single, orig_sr=initial_sr, target_sr=target_sr, res_type="polyphase"
                )

                target_length_resempled = x_single.shape[-1] * (target_sr // 2 // initial_sr)
                if len(x_resampled) > target_length_resempled:
                    x_resampled = x_resampled[:target_length_resempled]
                
                resampled_audio.append(x_resampled)
            x_reference = np.stack(resampled_audio)
            x_reference = torch.tensor(x_reference, dtype=padded_x.dtype).to(x_lr.device)

        current_reference_len = x_reference.shape[-1]
        pad_reference_len = expected_reference_len - current_reference_len

        padded_reference = torch.nn.functional.pad(x_reference, (0, pad_reference_len)).to(x_lr.device)
        padded_x_hr = torch.nn.functional.pad(x_hr, (0, expected_reference_len - x_hr.size(-1))).to(x_hr.device)

        x_res = padded_x_hr

        # x_res = self.encoder(x_res)
        x_res = self.get_stft(x_res, sampling_rate=target_sr).abs()

        latent = x_res.clone()

        x_res, _ = self.hifi(x_res)
    
        if self.use_waveunet:
            x_res = self.apply_waveunet_a2a(x_res, padded_reference)

        if self.use_spectralmasknet:
            x_res = self.apply_spectralmasknet(x_res)

        x_res = self.conv_post(x_res)
        x_res = torch.tanh(x_res)

        return x_res[..., :target_size], latent
