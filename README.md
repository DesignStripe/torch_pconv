# Torch Pconv

[![PyPI version](https://badge.fury.io/py/torch-pconv.svg)](https://badge.fury.io/py/torch-pconv)

Faster and more memory efficient implementation of the Partial Convolution 2D layer in PyTorch equivalent to the
standard Nvidia implementation.

This implementation has numerous advantages:

1. It is **strictly equivalent** in computation
   to [the reference implementation by Nvidia](https://github.com/NVIDIA/partialconv/blob/610d373f35257887d45adae84c86d0ce7ad808ec/models/partialconv2d.py)
   . I made unit tests to assess that all throughout development.
2. It's commented and more readable
3. It's faster and more memory efficient, which means you can use more layers on smaller GPUs. It's a good thing
   considering today's GPU prices.
4. It's a PyPI-published library. You can `pip` install it instead of copy/pasting source code, and get the benefit of (
   free) bugfixes when someone notice a bug in the implementation.

![Total memory cost (in bytes)](doc/2021-08-13_perfs.png?raw=true)

## Getting started

```sh
pip3 install torch_pconv
```

## Usage

```python3
import torch
from torch_pconv import PConv2d

images = torch.rand(32, 3, 256, 256)
masks = (torch.rand(32, 256, 256) > 0.5).to(torch.float32)

pconv = PConv2d(
    in_channels=3,
    out_channels=64,
    kernel_size=7,
    stride=1,
    padding=2,
    dilation=2,
    bias=True
)

output, shrunk_masks = pconv(images, masks)
```

## Performance improvement

### Test

You can
find [the reference implementation by Nvidia here](https://github.com/NVIDIA/partialconv/blob/610d373f35257887d45adae84c86d0ce7ad808ec/models/partialconv2d.py)
.

I tested their implementation vs mine one the following configuration:
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>in_channels</td><td>64</td></tr>
<tr><td>out_channels</td><td>128</td></tr>
<tr><td>kernel_size</td><td>9</td></tr>
<tr><td>stride</td><td>1</td></tr>
<tr><td>padding</td><td>3</td></tr>
<tr><td>bias</td><td>True</td></tr>
<tr><td>input height/width</td><td>256</td></tr>
</table>

The goal here was to produce the most computationally expensive partial convolution operator so that the performance
difference is displayed better.

I compute both the forward and the backward pass, in case one consumes more memory than the other.

### Results

![Total memory cost (in bytes)](doc/2021-08-13_perfs.png?raw=true)
<table>
<tr><th></th><th><pre>torch_pconv</pre></th><th>Nvidia® (Guilin)</th></tr>
<tr><td>Forward only</td><td><b>813 466 624</b></td><td>4 228 120 576</td></tr>
<tr><td>Backward only</td><td>1 588 201 480</td><td>1 588 201 480</td></tr>
<tr><td>Forward + Backward</td><td>2 405 797 640</td><td>6 084 757 512</td></tr>
</table>

## Development

To install the latest version from Github, run:

```
git clone git@github.com:DesignStripe/torch_pconv.git torch_pconv
cd torch_pconv
pip3 install -U .
```
