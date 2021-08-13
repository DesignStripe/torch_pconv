###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################
"""
Code by Guilin Liu at
https://github.com/NVIDIA/partialconv/blob/610d373f35257887d45adae84c86d0ce7ad808ec/models/partialconv2d.py

I tried to modify the least code: just enough to make is compatible with 3D masks (instead of 4D)
"""

import torch
import torch.nn.functional as F
from torch import nn


class PConvGuilin(nn.Conv2d):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        kwargs['multi_channel'] = True

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False

        self.return_mask = True

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                                                 self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
                             self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, inputs, mask_in=None):
        if len(inputs.shape) != 4 or len(mask_in.shape) != 3:
            raise TypeError()

        if inputs.dtype != torch.float32 or mask_in.dtype != torch.float32:
            raise TypeError()

        mask_in = mask_in[:, None].expand(-1, inputs.shape[1], -1, -1)

        if mask_in is not None or self.last_size != tuple(inputs.shape):
            self.last_size = tuple(inputs.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != inputs.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(inputs)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(inputs.data.shape[0], inputs.data.shape[1], inputs.data.shape[2],
                                          inputs.data.shape[3]).to(inputs)
                    else:
                        mask = torch.ones(1, 1, inputs.data.shape[2], inputs.data.shape[3]).to(inputs)
                else:
                    mask = mask_in

                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride,
                                            padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        raw_out = nn.Conv2d.forward(self, torch.mul(inputs, mask) if mask_in is not None else inputs)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask[:, 0]
        else:
            return output

    def set_weight(self, w):
        with torch.no_grad():
            self.weight.copy_(w)
        return self

    def set_bias(self, b):
        with torch.no_grad():
            self.bias.copy_(b)
        return self

    def get_weight(self):
        return self.weight

    def get_bias(self):
        return self.bias
