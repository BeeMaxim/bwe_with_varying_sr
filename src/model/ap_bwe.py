import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from src.utils.upsampling_utils import init_weights, get_padding
import numpy as np
import torchaudio
import librosa
import torchaudio.functional as aF
from typing import Tuple, List

LRELU_SLOPE = 0.1


def amp_pha_stft(audio, n_fft, hop_size, win_size, center=True):

    hann_window = torch.hann_window(win_size).to(audio.device)
    stft_spec = torch.stft(audio, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    log_amp = torch.log(torch.abs(stft_spec)+1e-4)
    pha = torch.angle(stft_spec)

    com = torch.stack((torch.exp(log_amp)*torch.cos(pha), 
                       torch.exp(log_amp)*torch.sin(pha)), dim=-1)

    return log_amp, pha, com


def amp_pha_istft(log_amp, pha, n_fft, hop_size, win_size, center=True):
    
    amp = torch.exp(log_amp)
    com = torch.complex(amp*torch.cos(pha), amp*torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    audio = torch.istft(com, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window, center=center)

    return audio



class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
        adanorm_num_embeddings (int, optional): Number of embeddings for AdaLayerNorm.
            None means non-conditional LayerNorm. Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        layer_scale_init_value= None,
        adanorm_num_embeddings = None,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.adanorm = adanorm_num_embeddings is not None
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, dim*3)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(dim*3, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x, cond_embedding_id = None):
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        if self.adanorm:
            assert cond_embedding_id is not None
            x = self.norm(x, cond_embedding_id)
        else:
            x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x


class APNet_BWE_Model(torch.nn.Module):
    def __init__(self, 
                 n_fft,
                 ConvNeXt_layers,
                 ConvNeXt_channels,
                 win_size,
                 hop_size):
        super(APNet_BWE_Model, self).__init__()
        self.adanorm_num_embeddings = None
        layer_scale_init_value =  1 / ConvNeXt_layers

        self.conv_pre_mag = nn.Conv1d(n_fft//2+1, ConvNeXt_channels, 7, 1, padding=get_padding(7, 1))
        self.norm_pre_mag = nn.LayerNorm(ConvNeXt_channels, eps=1e-6)
        self.conv_pre_pha = nn.Conv1d(n_fft//2+1, ConvNeXt_channels, 7, 1, padding=get_padding(7, 1))
        self.norm_pre_pha = nn.LayerNorm(ConvNeXt_channels, eps=1e-6)
        self.n_fft = n_fft
        self.win_size = win_size
        self.hop_size = hop_size

        self.convnext_mag = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=ConvNeXt_channels,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=self.adanorm_num_embeddings,
                )
                for _ in range(ConvNeXt_layers)
            ]
        )

        self.convnext_pha = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=ConvNeXt_channels,
                    layer_scale_init_value=layer_scale_init_value,
                    adanorm_num_embeddings=self.adanorm_num_embeddings,
                )
                for _ in range(ConvNeXt_layers)
            ]
        )

        self.norm_post_mag = nn.LayerNorm(ConvNeXt_channels, eps=1e-6)
        self.norm_post_pha = nn.LayerNorm(ConvNeXt_channels, eps=1e-6)
        self.apply(self._init_weights)
        self.linear_post_mag = nn.Linear(ConvNeXt_channels, n_fft//2+1)
        self.linear_post_pha_r = nn.Linear(ConvNeXt_channels, n_fft//2+1)
        self.linear_post_pha_i = nn.Linear(ConvNeXt_channels, n_fft//2+1)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, initial_sr, target_sr, **batch):
        audio_lr = aF.resample(batch["wav_hr"].double(), orig_freq=target_sr, new_freq=initial_sr)
        x_res = aF.resample(audio_lr, orig_freq=initial_sr, new_freq=target_sr)
        '''
        resampled_audio = []
        for i in range(x.size(0)):
            x_single = x[i].cpu().numpy()

            x_resampled = librosa.resample(x_single, orig_sr=initial_sr, target_sr=target_sr)
                
            resampled_audio.append(x_resampled)

        x_res = np.stack(resampled_audio)
        x_res = torch.tensor(x_res, dtype=x.dtype).to(x.device)'''
        #torchaudio.save(f'D:\hifi_ssr\\bwe_with_varying_sr\dems\\test\\init.wav', x_res[0].cpu(), target_sr)

        mag_nb, pha_nb, _ = amp_pha_stft(x_res.squeeze(1), self.n_fft, self.hop_size, self.win_size)

        x_mag = self.conv_pre_mag(mag_nb)
        x_pha = self.conv_pre_pha(pha_nb)
        x_mag = self.norm_pre_mag(x_mag.transpose(1, 2)).transpose(1, 2)
        x_pha = self.norm_pre_pha(x_pha.transpose(1, 2)).transpose(1, 2)

        for conv_block_mag, conv_block_pha in zip(self.convnext_mag, self.convnext_pha):
            x_mag = x_mag + x_pha
            x_pha = x_pha + x_mag
            x_mag = conv_block_mag(x_mag, cond_embedding_id=None)
            x_pha = conv_block_pha(x_pha, cond_embedding_id=None)

        x_mag = self.norm_post_mag(x_mag.transpose(1, 2))
        mag_wb = mag_nb + self.linear_post_mag(x_mag).transpose(1, 2)

        x_pha = self.norm_post_pha(x_pha.transpose(1, 2))
        x_pha_r = self.linear_post_pha_r(x_pha)
        x_pha_i = self.linear_post_pha_i(x_pha)
        pha_wb = torch.atan2(x_pha_i, x_pha_r).transpose(1, 2)

        com_wb = torch.stack((torch.exp(mag_wb)*torch.cos(pha_wb), 
                           torch.exp(mag_wb)*torch.sin(pha_wb)), dim=-1)
        
        audio_hr_g = amp_pha_istft(mag_wb, pha_wb, self.n_fft, self.hop_size, self.win_size).unsqueeze(1)
        #torchaudio.save(f'D:\hifi_ssr\\bwe_with_varying_sr\dems\\test\\final.wav', audio_hr_g[0].cpu(), target_sr)
        #print(audio_hr_g.shape)
        
        return audio_hr_g, [audio_hr_g]
    

class APBWE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.generator = APNet_BWE_Model(*args, **kwargs)

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.generator.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.generator.parameters() if p.requires_grad]
        )

        result_info = ""
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
