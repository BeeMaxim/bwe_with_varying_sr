import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from src.utils.upsampling_utils import init_weights, get_padding
from torch.nn import functional as F
from torch import sin, pow
from torch.nn import Parameter
from src.utils.upsampling_utils import TorchActivation1d, Snake, SnakeBeta


class AMPBlock1(torch.nn.Module):
    """
    AMPBlock applies Snake / SnakeBeta activation functions with trainable parameters that control periodicity, defined for each layer.
    AMPBlock1 has additional self.convs2 that contains additional Conv1d layers with a fixed dilation=1 followed by each layer in self.convs1

    Args:
        h (AttrDict): Hyperparameters.
        channels (int): Number of convolution channels.
        kernel_size (int): Size of the convolution kernel. Default is 3.
        dilation (tuple): Dilation rates for the convolutions. Each dilation layer has two convolutions. Default is (1, 3, 5).
        activation (str): Activation function type. Should be either 'snake' or 'snakebeta'. Default is None.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple = (1, 3, 5),
        activation: str = None,
    ):
        super().__init__()

        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=d,
                        padding=get_padding(kernel_size, d),
                    )
                )
                for d in dilation
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        stride=1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
                for _ in range(len(dilation))
            ]
        )
        self.convs2.apply(init_weights)

        self.num_layers = len(self.convs1) + len(
            self.convs2
        )  # Total number of conv layers

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if False and self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        # Activation functions
        if activation == "snake":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=Snake(
                            channels, alpha_logscale=True
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        elif activation == "snakebeta":
            self.activations = nn.ModuleList(
                [
                    Activation1d(
                        activation=SnakeBeta(
                            channels, alpha_logscale=True
                        )
                    )
                    for _ in range(self.num_layers)
                ]
            )
        else:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

    def forward(self, x):
        acts1, acts2 = self.activations[::2], self.activations[1::2]
        for c1, c2, a1, a2 in zip(self.convs1, self.convs2, acts1, acts2):
            xt = a1(x)
            xt = c1(xt)
            xt = a2(xt)
            xt = c2(xt)
            x = xt + x

        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)



class BigVUpsampling(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Select which Activation1d, lazy-load cuda version to ensure backward compatibility
        if False and self.h.get("use_cuda_kernel", False):
            from alias_free_activation.cuda.activation1d import (
                Activation1d as CudaActivation1d,
            )

            Activation1d = CudaActivation1d
        else:
            Activation1d = TorchActivation1d

        self.num_kernels = 3
        self.num_upsamples = 4

        # Pre-conv
        self.conv_pre = weight_norm(
            Conv1d(128, 128, 7, 1, padding=3)
        )

        # Define which AMPBlock to use. BigVGAN uses AMPBlock1 as default
        resblock_class = AMPBlock1

        # Transposed conv-based upsamplers. does not apply anti-aliasing
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip([8, 8, 2, 2], [16, 16, 4, 4])):
            self.ups.append(
                nn.ModuleList(
                    [
                        weight_norm(
                            ConvTranspose1d(
                                128 // (2**i),
                                128 // (2 ** (i + 1)),
                                k,
                                u,
                                padding=(k - u) // 2,
                            )
                        )
                    ]
                )
            )

        # Residual blocks using anti-aliased multi-periodicity composition modules (AMP)
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = 128 // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip([3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
            ):
                self.resblocks.append(
                    resblock_class(ch, k, d, activation="snakebeta")
                )

        # Post-conv
        activation_post = (
            Snake(ch, alpha_logscale=True)
            if False
            else (
                SnakeBeta(ch, alpha_logscale=True)
            )
        )
        if activation_post is None:
            raise NotImplementedError(
                "activation incorrectly specified. check the config file and look for 'activation'."
            )

        self.activation_post = Activation1d(activation=activation_post)

        # Whether to use bias for the final conv_post. Default to True for backward compatibility
        self.use_bias_at_final = True
        self.conv_post = weight_norm(
            Conv1d(ch, 1, 7, 1, padding=3, bias=self.use_bias_at_final)
        )

        # Weight initialization
        for i in range(len(self.ups)):
            self.ups[i].apply(init_weights)
        self.conv_post.apply(init_weights)

        # Final tanh activation. Defaults to True for backward compatibility
        self.use_tanh_at_final = True

    def forward(self, x):
        # Pre-conv
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            # Upsampling
            for i_up in range(len(self.ups[i])):
                x = self.ups[i][i_up](x)
            # AMP blocks
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        # Post-conv
        x = self.activation_post(x)
        # x = self.conv_post(x)
        # Final tanh activation
        '''
        if self.use_tanh_at_final:
            x = torch.tanh(x)
        else:
            x = torch.clamp(x, min=-1.0, max=1.0)  # Bound the output to [-1, 1]'''

        return x

    def remove_weight_norm(self):
        try:
            print("Removing weight norm...")
            for l in self.ups:
                for l_i in l:
                    remove_weight_norm(l_i)
            for l in self.resblocks:
                l.remove_weight_norm()
            remove_weight_norm(self.conv_pre)
            remove_weight_norm(self.conv_post)
        except ValueError:
            print("[INFO] Model already removed weight norm. Skipping!")
            pass