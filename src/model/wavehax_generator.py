from functools import partial
import torch.nn.functional as F

import torch
import numpy as np
import pyworld
from einops import rearrange
from torch import Tensor, nn


class NormLayer(nn.Module):
    def __init__(
        self, channels: int, eps = 1e-6, affine = True
    ) -> None:
        """
        Initialize the NormLayer module.

        Args:
            channels (int): Number of input features.
            eps (float, optional): A small constant added to the denominator for numerical stability (default: 1e-6).
            affine (bool, optional): If True, this module has learnable affine parameters (default: True).
        """
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.gamma = nn.Parameter(torch.ones(channels))
            self.beta = nn.Parameter(torch.zeros(channels))

    def normalize(
        self,
        x: Tensor,
        dim: int,
        mean = None,
        var = None,
    ):
        """
        Apply normalization to the input tensor.

        Args:
            x (Tensor): Input tensor with shape (batch, channels, ...).
            dim (int): Dimensions along which statistics are calculated.
            mean (Tensor, optional): Mean tensor (default: None).
            var (Tensor, optional): Variance tensor (default: None).

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Normalized tensor and statistics.
        """
        # Calculate the mean along dimensions to be reduced
        if mean is None:
            mean = x.mean(dim, keepdim=True)

        # Centerize the input tensor
        x = x - mean

        # Calculate the variance
        if var is None:
            var = (x**2).mean(dim=dim, keepdim=True)

        # Normalize
        x = x / torch.sqrt(var + self.eps)

        if self.affine:
            shape = [1, self.channels] + [1] * (x.ndim - 2)
            x = self.gamma.view(*shape) * x + self.beta.view(*shape)

        return x, mean, var


class LayerNorm1d(NormLayer):
    def __init__(
        self,
        channels: int,
        framewise: bool = False,
        eps = 1e-6,
        affine = True,
    ) -> None:
        """
        Initialize the LayerNorm1d module.

        Args:
            channels (int): Number of input features.
            eps (float, optional): A small constant added to the denominator for numerical stability (default: 1e-6).
            affine (bool, optional): If True, this module has learnable affine parameters (default: True).
        """
        super().__init__(channels, eps, affine)
        if framewise:
            self.reduced_dim = [1]
        else:
            self.reduced_dim = [1, 2]

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply layer normalization to the input tensor.

        Args:
            x (Tensor): Input tensor with shape (batch, channels, height, width).

        Returns:
            Tensor: Normalized tensor.
        """
        x, *_ = self.normalize(x, dim=self.reduced_dim)
        return x


class LayerNorm2d(NormLayer):
    def __init__(
        self,
        channels: int,
        framewise: bool = False,
        eps = 1e-6,
        affine = True,
    ) -> None:
        """
        Initialize the LayerNorm2d module.

        Args:
            channels (int): Number of input features.
            framewise (bool, optional): If True, normalization is performed independently for each time frame.
            eps (float, optional): A small constant added to the denominator for numerical stability (default: 1e-6).
            affine (bool, optional): If True, this module has learnable affine parameters (default: True).
        """
        super().__init__(channels, eps, affine)
        if framewise:
            self.reduced_dim = [1, 2]
        else:
            self.reduced_dim = [1, 2, 3]

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply normalization to the input tensor.

        Args:
            x (Tensor): Input tensor with shape (batch, channels, height, width).

        Returns:
            Tensor: Normalized tensor.
        """
        x, *_ = self.normalize(x, dim=self.reduced_dim)
        return x
    

class BatchNorm1d(NormLayer):
    def __init__(
        self,
        channels: int,
        eps = 1e-6,
        affine = True,
        momentum = 0.1,
        track_running_stats = True,
    ) -> None:
        """
        Initialize the BatchNorm1d module.

        Args:
            channels (int): Number of input features.
            eps (float, optional): A small constant added to the denominator for numerical stability (default: 1e-6).
            affine (bool, optional): If True, this module has learnable affine parameters (default: True).
            momentum (float, optional): The value used for the running_mean and running_var computation.
                Can be set to None for cumulative moving average, i.e. simple average (default: None).
            track_running_stats (bool, optional): If True, tracks running mean and variance during training.
        """
        super().__init__(channels, eps, affine)
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        if track_running_stats:
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
            self.register_buffer("running_mean", torch.zeros(1, channels, 1))
            self.register_buffer("running_var", torch.ones(1, channels, 1))
        self.reduced_dim = [0, 2]

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply batch normalization to the input tensor.

        Args:
            x (Tensor): Input tensor with shape (batch, channels, height, width).

        Returns:
            Tensor: Normalized tensor.
        """
        # Get the running statistics if needed
        if (not self.training) and self.track_running_stats:
            mean = self.running_mean
            var = self.running_var
        else:
            mean = var = None

        x, mean, var = self.normalize(x, dim=self.reduced_dim, mean=mean, var=var)

        # Update the running statistics
        if self.training and self.track_running_stats:
            with torch.no_grad():
                # Update the number of tracked samples
                self.num_batches_tracked += 1

                # Get the weight for cumulative or exponential moving average
                if self.momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:
                    exponential_average_factor = self.momentum

                # Update the running mean and covariance matrix
                self.running_mean = (
                    exponential_average_factor * mean
                    + (1 - exponential_average_factor) * self.running_mean
                )
                n = x.numel() / x.size(1)
                self.running_var = (
                    exponential_average_factor * var * n / (n - 1)
                    + (1 - exponential_average_factor) * self.running_var
                )

        return x
    

class BatchNorm2d(BatchNorm1d):
    def __init__(
        self,
        channels: int,
        eps = 1e-6,
        affine = True,
        momentum = 0.1,
        track_running_stats = True,
    ) -> None:
        """
        Initialize the BatchNorm2d module.

        Args:
            channels (int): Number of input features.
            eps (float, optional): A small constant added to the denominator for numerical stability (default: 1e-6).
            affine (bool, optional): If True, this module has learnable affine parameters (default: True).
            momentum (float, optional): The value used for the running_mean and running_var computation.
                Can be set to None for cumulative moving average, i.e. simple average (default: None).
            track_running_stats (bool, optional): If True, tracks running mean and variance during training.
        """
        super().__init__(channels, eps, affine)
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        if track_running_stats:
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
            self.register_buffer("running_mean", torch.zeros(1, channels, 1, 1))
            self.register_buffer("running_var", torch.ones(1, channels, 1, 1))
        self.reduced_dim = [0, 2, 3]

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply batch normalization to the input tensor.

        Args:
            x (Tensor): Input tensor with shape (batch, channels, height, width).

        Returns:
            Tensor: Normalized tensor.
        """
        return super().forward(x)
    

def drop_path(
    x: Tensor,
    drop_prob = 0.0,
    training = False,
    scale_by_keep = True,
) -> Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of drop_prob in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(
        self, drop_prob = 0.0, scale_by_keep = True
    ) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: Tensor) -> Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f"drop_prob={round(self.drop_prob,3):0.3f}"
    

class ConvNeXtBlock2d(nn.Module):
    """
    A 2D residual block module based on ConvNeXt architecture.

    Reference:
        - https://github.com/facebookresearch/ConvNeXt
    """

    def __init__(
        self,
        channels: int,
        mult_channels: int,
        kernel_size: int,
        drop_prob: float = 0.0,
        use_layer_norm: bool = True,
        framewise_norm: bool = True,
        layer_scale_init_value: float = None,
    ) -> None:
        """
        Initialize the ConvNeXtBlock2d module.

        Args:
            channels (int): Number of input and output channels for the block.
            mult_channels (int): Channel expansion factor used in pointwise convolutions.
            kernel_size (int): Size of the depthwise convolution kernel.
            drop_prob (float, optional): Probability of dropping paths for stochastic depth (default: 0.0).
            use_layer_norm (bool, optional): If True, layer normalization is used; otherwise,
                batch normalization is applied (default: True).
            framewise_norm (bool, optional): If True, normalization is performed independently for each time frame.
            layer_scale_init_value (float, optional): Initial value for the learnable layer scale parameter.
                If None, no scaling is applied (default: None).
        """
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert kernel_size[0] % 2 == 1, "Kernel size must be odd number."
        assert kernel_size[1] % 2 == 1, "Kernel size must be odd number."
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=(kernel_size[0] // 2, kernel_size[1] // 2),
            groups=channels,
            bias=False,
            padding_mode="reflect",
        )
        if use_layer_norm:
            self.norm = LayerNorm2d(channels, framewise=framewise_norm)
        else:
            self.norm = BatchNorm2d(channels)
        self.pwconv1 = nn.Conv2d(channels, channels * mult_channels, 1)
        self.nonlinear = nn.GELU()
        self.pwconv2 = nn.Conv2d(channels * mult_channels, channels, 1)
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones(1, channels, 1, 1),
                requires_grad=True,
            )
            if layer_scale_init_value is not None
            else None
        )
        self.drop_path = DropPath(drop_prob)

    def forward(self, x: Tensor) -> Tensor:
        """
        Calculate forward propagation.

        Args:
            x (Tensor): Input tensor with shape (batch, channels, height, width).

        Returns:
            Tensor: Output tensor of the same shape (batch, channels, height, width).
        """
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.nonlinear(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = residual + self.drop_path(x)
        return x


def to_log_magnitude_and_phase(
    real: Tensor, imag: Tensor, clip_value = 1e-10
):
    """
    Convert real and imaginary components of a complex signal to log-magnitude and phase.

    Args:
        real (Tensor): Real part of the complex signal.
        imag (Tensor): Imaginary part of the complex signal.
        clip_value (float, optional): Minimum value for magnitude to avoid log of zero (default: 1e-10).

    Returns:
        Tuple[Tensor, Tensor]: Log-magnitude and phase of the input complex signal.
    """
    magnitude = torch.sqrt(torch.clamp(real**2 + imag**2, min=clip_value))
    log_magnitude = torch.log(magnitude)
    phase = torch.atan2(imag, real)
    return log_magnitude, phase


def to_real_imaginary(
    log_magnitude: Tensor, phase: Tensor, clip_value = 1e2
):
    """
    Convert log-magnitude and implicit phase wrapping back to real and imaginary components of a complex signal.

    Args:
        log_magnitude (Tensor): Log-magnitude of the complex signal.
        phase (Tensor): Implicit phase wrapping spectra as in Vocos.
        clip_value (float, optional): Maximum allowed value for magnitude after exponentiation (default: 1e2).

    Returns:
        Tuple[Tensor, Tensor]: Real and imaginary components of the complex signal.

    References:
        - https://arxiv.org/abs/2306.00814
        - https://github.com/gemelo-ai/vocos
    """
    magnitude = torch.clip(torch.exp(log_magnitude), max=clip_value)
    real, imag = magnitude * torch.cos(phase), magnitude * torch.sin(phase)
    return real, imag


class STFT(nn.Module):
    """
    Short-Time Fourier Transform (STFT) module.

    References:
        - https://github.com/gemelo-ai/vocos
        - https://github.com/echocatzh/torch-mfcc
    """

    def __init__(
        self, n_fft: int, hop_length: int, window = "hann_window"
    ) -> None:
        """
        Initialize the STFT module.

        Args:
            n_fft (int): Number of Fourier transform points (FFT size).
            hop_length (int): Hop length (frameshift) in samples.
            window (str, optional): Name of the window function (default: "hann_window").
        """
        super().__init__()
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.hop_length = hop_length

        # Create the window function and its squared values for normalization
        window = getattr(torch, window)(self.n_fft).reshape(1, n_fft, 1)
        self.register_buffer("window", window.reshape(1, n_fft, 1))
        window_envelope = window.square()
        self.register_buffer("window_envelope", window_envelope.reshape(1, n_fft, 1))

        # Create the kernel for enframe operation (sliding windows)
        enframe_kernel = torch.eye(self.n_fft).unsqueeze(1)
        self.register_buffer("enframe_kernel", enframe_kernel)

    def forward(self, x: Tensor, norm = None):
        """
        Perform the forward Short-Time Fourier Transform (STFT) on the input waveform.

        Args:
            x (Tensor): Input waveform with shape (batch, samples) or (batch, 1, samples).
            norm (str, optional): Normalization mode for the FFT (default: None).

        Returns:
            Tuple[Tensor, Tensor]: Real and imaginary parts of the STFT result.
        """
        # Apply zero-padding to the input signal
        pad = self.n_fft - self.hop_length
        pad_left = pad // 2
        x = F.pad(x, (pad_left, pad - pad_left))

        # Enframe the padded waveform (sliding windows)
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = F.conv1d(x, self.enframe_kernel, stride=self.hop_length)

        # Perform the forward real-valued DFT on each frame
        x = x * self.window
        x_stft = torch.fft.rfft(x, dim=1, norm=norm)
        real, imag = x_stft.real, x_stft.imag

        return real, imag

    def inverse(self, real: Tensor, imag: Tensor, norm = None) -> Tensor:
        """
        Perform the inverse Short-Time Fourier Transform (iSTFT) to reconstruct the waveform from the complex spectrogram.

        Args:
            real (Tensor): Real part of the complex spectrogram with shape (batch, n_bins, frames).
            imag (Tensor): Imaginary part of the complex spectrogram with shape (batch, n_bins, frames).
            norm (str, optional): Normalization mode for the inverse FFT (default: None).

        Returns:
            Tensor: Reconstructed waveform with shape (batch, 1, samples).
        """
        # Validate shape and dimensionality
        assert real.shape == imag.shape and real.ndim == 3

        # Ensure the input represents a one-sided spectrogram
        assert real.size(1) == self.n_bins

        frames = real.shape[2]
        samples = frames * self.hop_length

        # Inverse RDFT and apply windowing, followed by overlap-add
        print("!!!", real.shape)
        x = torch.fft.irfft(torch.complex(real, imag), dim=1, norm=norm)
        '''
        x  = torch.istft(
                torch.complex(real, imag),
                n_fft,
                hop_length=hop_size,
                win_length=win_size,
                window=hann_window[str(y.device)],
                center=center,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )'''
        x = x * self.window
        x = F.conv_transpose1d(x, self.enframe_kernel, stride=self.hop_length)

        # Compute window envelope for normalization
        window_envelope = F.conv_transpose1d(
            self.window_envelope.repeat(1, 1, frames),
            self.enframe_kernel,
            stride=self.hop_length,
        )

        # Remove padding
        pad = (self.n_fft - self.hop_length) // 2
        x = x[..., pad : samples + pad]
        window_envelope = window_envelope[..., pad : samples + pad]

        # Normalize the output by the window envelope
        assert (window_envelope > 1e-11).all()
        x = x / window_envelope

        return x
    

def generate_pcph_closed_form(
    f0: Tensor,
    hop_length: int,
    sample_rate: int,
    noise_amplitude = 0.01,
    random_init_phase = True,
    power_factor = 0.1,
    max_frequency = None,
    epsilon = 1e-6,
    use_modulo = True
) -> torch.Tensor:
    """
    An optimized O(1) generator for Pseudo-Constant-Power Harmonic waveforms.
    Uses the Dirichlet kernel closed-form identity formula for speed and efficiency.
    """
    batch, _, frames = f0.size()
    device = f0.device

# F0 upsampling

# optionally you could use pchip_upsampler I've prepared. check modules
# You'd need to import it and simply:
#     pchip_f0_upsampler = PchipF0UpsamplerTorch(scale_factor=hop_length)
#     f0_upsampled = pchip_f0_upsampler(f0)

    f0_upsampled = F.interpolate(
        f0, scale_factor=hop_length, mode='linear', align_corners=False
    )

    # Preparation
    total_length = f0_upsampled.shape[-1]
    noise = torch.randn((batch, 1, total_length), device=device) * noise_amplitude
    # Return early on silent samples
    if torch.all(f0 == 0.0):
        return noise

    # Calculate Phase (Theta)
    # phase = 2 * pi * integral(f0 / sr)
    phase_increment = f0_upsampled / sample_rate

    # Randomize initial phase
    if random_init_phase:
        init_phase = torch.rand((batch, 1, 1), device=device)
        # phase_increment[:, :, :1] = phase_increment[:, :, :1] + init_phase # Out of place
        phase_increment[:, :, :1] += init_phase # In-place

    # Cumsum
    # Multiplying by 2pi at the end to save ops during the cumsum
    phase = torch.cumsum(phase_increment.double(), dim=2) * 2.0 * torch.pi
    if use_modulo:
        phase = torch.fmod(phase, 2.0 * torch.pi)
    phase = phase.float()

    # Dynamic harmonic count (N)
    # N is the max harmonic index before aliasing (Nyquist)
    # N(t) = floor( MaxFreq / f0(t) )
    nyquist = sample_rate / 2.0
    limit_freq = max_frequency if max_frequency is not None else nyquist

    # Zero-Division safety for unvoiced segments
    safe_f0 = torch.clamp(f0_upsampled, min=1e-5)
    N = torch.floor(limit_freq / safe_f0)

    # Closed-Form Summation
    # Sum(sin(k*theta)) = (cos(theta/2) - cos((N + 0.5)*theta)) / (2*sin(theta/2))

    half_phase = phase / 2.0
    # Numerator: cos(theta/2) - cos((N + 0.5)theta)
    numerator = torch.cos(half_phase) - torch.cos((N + 0.5) * phase)

    # Denominator: 2 * sin(theta/2)
    # We need a safe division because sin(theta/2) is 0 at phase = 0, 2pi, etc.
    denominator = 2.0 * torch.sin(half_phase)

    # Safe Division:
    # Where denominator is close to 0, the theoretical limit of the sum is 0 (for sine sum).
    # We use a mask to avoid NaNs.
    # Note: For Sum of Cosines (Dirichlet), the limit is N. For Sum of Sines, it is 0.
    not_singular = torch.abs(denominator) > epsilon

    # Initialize harmonics container
    harmonics = torch.zeros_like(phase)

    # Calculate only where stable
    harmonics[not_singular] = numerator[not_singular] / denominator[not_singular]
    # (Where singular, we leave as 0.0, which is correct for sum of sines at phase 0)

    # Amplitude Normalization (Pseudo-Constant-Power)
    # We calculate this dynamically per sample based on N
    # Mask out silence/unvoiced regions (where f0 was 0)
    vuv_mask = (f0_upsampled > 0).float()

    # Power Factor Normalization: amp = P * sqrt(2/N)
    # We clamp N to 1.0 to prevent sqrt(div/0)
    amp_scale = power_factor * torch.sqrt(2.0 / torch.clamp(N, min=1.0))

    # Apply masks
    prior_signal = (harmonics * amp_scale * vuv_mask) + noise

    return prior_signal


class WavehaxGenerator(nn.Module):
    """
    Wavehax generator module.

    This module produces time-domain waveforms through complex spectrogram estimation
    based on the integration of 2D convolution and harmonic prior spectrograms.
    """

    def __init__(
        self,
        in_channels: int,
        channels: int,
        mult_channels: int,
        kernel_size: int,
        num_blocks: int,
        n_fft: int,
        hop_length: int,
        sample_rate: int,
        drop_prob: float = 0.0,
        use_layer_norm: bool = True,
        framewise_norm: bool = False,
        use_logmag_phase: bool = False,
    ) -> None:
        """
        Initialize the WavehaxGenerator module.

        Args:
            in_channels (int): Number of conditioning feature channels.
            channels (int): Number of hidden feature channels.
            mult_channels (int): Channel expansion multiplier for ConvNeXt blocks.
            kernel_size (int): Kernel size for ConvNeXt blocks.
            num_blocks (int): Number of ConvNeXt residual blocks.
            n_fft (int): Number of Fourier transform points (FFT size).
            hop_length (int): Hop length (frameshift) in samples.
            sample_rate (int): Sampling frequency of input and output waveforms in Hz.
            prior_type (str): Type of prior waveforms used.
            drop_prob (float): Probability of dropping paths for stochastic depth (default: 0.0).
            use_layer_norm (bool): If True, layer normalization is used; otherwise,
                batch normalization is applied (default: True).
            use_logmag_phase (bool): Whether to use log-magnitude and phase for STFT (default: False).
        """
        super().__init__()
        self.in_channels = in_channels
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.use_logmag_phase = use_logmag_phase

        # Prior waveform generator
        self.prior_generator = partial(
            generate_pcph_closed_form,
            hop_length=self.hop_length,
            sample_rate=sample_rate,
        )

        # STFT layer
        self.stft = STFT(n_fft=n_fft, hop_length=hop_length)

        # Input projection layers
        n_bins = n_fft // 2 + 1
        self.prior_proj = nn.Conv1d(
            n_bins, n_bins, 7, padding=3, padding_mode="reflect"
        )
        self.cond_proj = nn.Conv1d(
            in_channels, n_bins, 7, padding=3, padding_mode="reflect"
        )

        # Input normalization and projection layers
        self.input_proj = nn.Conv2d(5, channels, 1, bias=False)
        self.input_norm = LayerNorm2d(channels, framewise=framewise_norm)

        # ConvNeXt-based residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = ConvNeXtBlock2d(
                channels,
                mult_channels,
                kernel_size,
                drop_prob=drop_prob,
                use_layer_norm=use_layer_norm,
                framewise_norm=framewise_norm,
                layer_scale_init_value=1 / num_blocks,
            )
            self.blocks += [block]

        # Output normalization and projection layers
        self.output_norm = LayerNorm2d(channels, framewise=framewise_norm)
        self.output_proj = nn.Conv2d(channels, 2, 1)

        self.apply(self.init_weights)

    def init_weights(self, m) -> None:
        """
        Initialize weights of the module.

        Args:
            m (Any): Module to initialize.
        """
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(self, cond: Tensor, padded_reference: Tensor) -> Tensor:
        """
        Calculate forward propagation.

        Args:
            cond (Tensor): Conditioning features with shape (batch, in_channels, frames).
            f0 (Tensor): F0 sequences with shape (batch, 1, frames).

        Returns:
            Tensor: Generated waveforms with shape (batch, 1, frames * hop_length).
            Tensor: Generated prior waveforms with shape (batch, 1, frames * hop_length).
        """
        print(cond.shape)
        print(padded_reference.shape)
        f0_list = []
        for _ in range(padded_reference.size(0)):
            f0 = pyworld.harvest(
                padded_reference[0].squeeze(0).cpu().detach().numpy().astype(np.float64),
                fs=self.sample_rate,
                f0_floor=100,
                f0_ceil=1000,
                frame_period=1000 * 256 / self.sample_rate,
            )[0]
            print(f0.shape)

            f0_list.append(f0)

        f0 = np.stack(f0_list)
        f0 = torch.tensor(f0, dtype=padded_reference.dtype).to(padded_reference.device).unsqueeze(1)[..., :cond.size(-1)]
        #cond = F.pad(cond, (0, f0.size(-1) - cond.size(-1))).to(cond.device)
        print(f0.shape, cond.shape)

        # Generate prior waveform and compute spectrogram
        with torch.no_grad():
            prior = self.prior_generator(f0)
            real, imag = self.stft(prior)
            if self.use_logmag_phase:
                prior1, prior2 = to_log_magnitude_and_phase(real, imag)
            else:
                prior1, prior2 = real, imag

        # Apply input projection
        prior1_proj = self.prior_proj(prior1)
        prior2_proj = self.prior_proj(prior2)
        cond = self.cond_proj(cond)

        print(prior1.shape, prior1_proj.shape, cond.shape)

        # Convert to 2d representation
        x = torch.stack([prior1, prior2, prior1_proj, prior2_proj, cond], dim=1)
        x = self.input_proj(x)
        x = self.input_norm(x)

        # Apply residual blocks
        for f in self.blocks:
            x = f(x)

        # Apply output projection
        x = self.output_norm(x)
        x = self.output_proj(x)

        # Apply iSTFT followed by overlap and add
        if self.use_logmag_phase:
            real, imag = to_real_imaginary(x[:, 0], x[:, 1])
        else:
            real, imag = x[:, 0], x[:, 1]
        x = self.stft.inverse(real, imag)

        x_res = torch.zeros((x.size(0), 8, x.size(-1))).to(x.device)
        x_res[:] = x

        return x_res, prior

    @torch.inference_mode()
    def inference(self, cond: Tensor, f0: Tensor) -> Tensor:
        return self(cond, f0)[0]
