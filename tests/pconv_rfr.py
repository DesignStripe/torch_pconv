###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################
"""
Code by Guilin by but probably with some modifications by jingyuanli001, code at
https://github.com/jingyuanli001/RFR-Inpainting/blob/faed6f154e01fc3accce5dff82a5b28e6f426fbe/modules/partialconv2d.py

I tried to modify the least code: just enough to make is compatible with 3D masks (instead of 4D)
"""

import torch
import torch.nn.functional as F
from torch import nn


class PConvRFR(nn.Conv2d):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.ones = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0],
                               self.kernel_size[1])

        # max value of one convolution's window
        self.slide_winsize = self.ones.shape[1] * self.ones.shape[2] * self.ones.shape[3]

        self.update_mask = None
        self.mask_ratio = None

    def forward(self, inputs, mask=None):
        if len(inputs.shape) != 4 or len(mask.shape) != 3:
            raise TypeError()

        if inputs.dtype != torch.float32 or mask.dtype != torch.float32:
            raise TypeError()

        mask = mask[:, None].expand(-1, inputs.shape[1], -1, -1)

        with torch.no_grad():
            self.update_mask = F.conv2d(mask, self.ones.to(mask), bias=None, stride=self.stride,
                                        padding=self.padding,
                                        dilation=self.dilation,
                                        groups=1)

            self.mask_ratio = self.slide_winsize / (self.update_mask + 1e-8)
            self.update_mask = torch.clamp(self.update_mask, 0, 1)
            self.mask_ratio *= self.update_mask

        raw_out = nn.Conv2d.forward(self, inputs * mask)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = raw_out * self.mask_ratio

        return output, self.update_mask[:, 0]

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
