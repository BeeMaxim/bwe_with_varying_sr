from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Conv1d
from torch.nn.utils import weight_norm
from torch.nn.utils import spectral_norm
from typing import List

from src.model.pqmf import PQMF
from src.utils.upsampling_utils import get_padding


class CoMBDBlock(torch.nn.Module):
    def __init__(
        self,
        h_u: List[int],
        d_k: List[int],
        d_s: List[int],
        d_d: List[int],
        d_g: List[int],
        d_p: List[int],
        op_f: int,
        op_k: int,
        op_g: int,
        use_spectral_norm=False
    ):
        super(CoMBDBlock, self).__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm

        self.convs = nn.ModuleList()
        filters = [[1, h_u[0]]]
        for i in range(len(h_u) - 1):
            filters.append([h_u[i], h_u[i + 1]])
        for _f, _k, _s, _d, _g, _p in zip(filters, d_k, d_s, d_d, d_g, d_p):
            self.convs.append(norm_f(
                Conv1d(
                    in_channels=_f[0],
                    out_channels=_f[1],
                    kernel_size=_k,
                    stride=_s,
                    dilation=_d,
                    groups=_g,
                    padding=_p
                )
            ))
        self.projection_conv = norm_f(
            Conv1d(
                in_channels=filters[-1][1],
                out_channels=op_f,
                kernel_size=op_k,
                groups=op_g
            )
        )

    def forward(self, x):
        fmap = []
        for block in self.convs:
            x = block(x)
            x = F.leaky_relu(x, 0.2)
            fmap.append(x)
        x = self.projection_conv(x)
        return x, fmap


class CoMBD(torch.nn.Module):
    def __init__(self, 
                 h_u: List[int],
                 d_k: List[int],
                 d_s: List[int],
                 d_d: List[int],
                 d_g: List[int],
                 d_p: List[int],
                 op_f: int,
                 op_k: int,
                 op_g: int, 
                 pqmf_config,
                 pqmf_list=None, use_spectral_norm=False):
        super(CoMBD, self).__init__()

        if pqmf_list is not None:
            self.pqmf = pqmf_list
        else:
            self.pqmf = [
                PQMF(*pqmf_config["lv2"]),
                PQMF(*pqmf_config["lv1"])
            ]
            self.pqmf_2 = nn.ModuleList([
                PQMF(*pqmf_config["lv2"]),
                PQMF(*pqmf_config["lv1"])
            ])
            for param in self.pqmf_2.parameters():
                param.requires_grad = False

        self.blocks = nn.ModuleList()
        for _h_u, _d_k, _d_s, _d_d, _d_g, _d_p, _op_f, _op_k, _op_g in zip(
            h_u,
            d_k,
            d_s,
            d_d,
            d_g,
            d_p,
            op_f,
            op_k,
            op_g,
        ):
            self.blocks.append(CoMBDBlock(
                _h_u,
                _d_k,
                _d_s,
                _d_d,
                _d_g,
                _d_p,
                _op_f,
                _op_k,
                _op_g,
            ))

    def _block_forward(self, input, blocks, outs, f_maps):
        for x, block in zip(input, blocks):
            out, f_map = block(x)
            outs.append(out)
            f_maps.append(f_map)
        return outs, f_maps

    def _pqmf_forward(self, ys, ys_hat):
        # preprocess for multi_scale forward
        multi_scale_inputs = []
        multi_scale_inputs_hat = []
        for pqmf in self.pqmf:
            multi_scale_inputs.append(
                pqmf.to(ys[-1]).analysis(ys[-1])[:, :1, :]
            )
            multi_scale_inputs_hat.append(
                pqmf.to(ys[-1]).analysis(ys_hat[-1])[:, :1, :]
            )

        outs_real = []
        f_maps_real = []
        # real
        # for hierarchical forward
        outs_real, f_maps_real = self._block_forward(
            ys, self.blocks, outs_real, f_maps_real)
        # for multi_scale forward
        outs_real, f_maps_real = self._block_forward(
            multi_scale_inputs, self.blocks[:-1], outs_real, f_maps_real)

        outs_fake = []
        f_maps_fake = []
        # predicted
        # for hierarchical forward
        outs_fake, f_maps_fake = self._block_forward(
            ys_hat, self.blocks, outs_fake, f_maps_fake)
        # for multi_scale forward
        outs_fake, f_maps_fake = self._block_forward(
            multi_scale_inputs_hat, self.blocks[:-1], outs_fake, f_maps_fake)

        return outs_real, outs_fake, f_maps_real, f_maps_fake

    def forward(self, x_gt, x_fake, generator_outs, **batch):
        padded_size = generator_outs[-1].size(-1) * 2 - x_gt.size(-1)

        ys = [
            self.pqmf_2[0].analysis(
                F.pad(x_gt, (0, padded_size))
            )[:, :1],
            self.pqmf_2[1].analysis(
                F.pad(x_gt, (0, padded_size))
            )[:, :1],
            x_gt
        ]
        outs_real, outs_fake, f_maps_real, f_maps_fake = self._pqmf_forward(
            ys, generator_outs + [x_fake])

        return outs_real, f_maps_real, outs_fake, f_maps_fake
