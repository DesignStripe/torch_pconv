import itertools
import unittest
from functools import partial
from typing import List, Type, Dict, Tuple, Callable, Union, Iterable

import torch
from pshape import pshape
from torch import Tensor
from torch.profiler import profile, ProfilerActivity

from torch_pconv import PConv2d
from pconv_guilin import PConvGuilin
from pconv_rfr import PConvRFR
from conv_config import ConvConfig

PConvLike = torch.nn.Module


class TestPConv(unittest.TestCase):
    pconv_classes = [
        PConvGuilin,
        PConvRFR,
        # This forces numerical error to be the same as other implementations, but makes the computation a bit slower
        partial(PConv2d, legacy_behaviour=True),
    ]

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def test_output_shapes(self):
        b, c, h = 16, 3, 256
        image, mask = self.mkinput(b=b, c=c, h=h)
        configs = [
            ConvConfig(3, 64, 5, padding=2, stride=2),
            ConvConfig(64, 64, 5, padding=1),
            ConvConfig(64, 64, 3, padding=4),
            ConvConfig(64, 64, 7, padding=5),
            ConvConfig(64, 32, 3, padding=2),
        ]
        expected_heights = (128, 126, 132, 136, 138,)

        self.assertEqual(len(configs), len(expected_heights))

        outputs_imgs, outputs_masks = image, mask
        for expected_height, config in zip(expected_heights, configs):
            outputs_imgs, outputs_masks = self.run_pconvs(self.pconv_classes, config=config)(outputs_imgs,
                                                                                             outputs_masks)
            for clazz in self.pconv_classes:
                img, mask = outputs_imgs[clazz], outputs_masks[clazz]
                self.assertTupleEqual(tuple(img.shape), (b, config.out_channels, expected_height, expected_height))
                self.assertTupleEqual(tuple(mask.shape), (b, expected_height, expected_height))

    def test_output_dtype(self):
        b, c, h = 16, 3, 256
        image, mask = self.mkinput(b=b, c=c, h=h)
        configs = [
            ConvConfig(3, 64, 5, padding=2, stride=2),
            ConvConfig(64, 64, 5, padding=1),
            ConvConfig(64, 64, 3, padding=4),
            ConvConfig(64, 64, 7, padding=5),
            ConvConfig(64, 32, 3, padding=2),
        ]
        expected_heights = (128, 126, 132, 136, 138,)

        self.assertEqual(len(configs), len(expected_heights))

        outputs_imgs, outputs_masks = image, mask
        for expected_height, config in zip(expected_heights, configs):
            outputs_imgs, outputs_masks = self.run_pconvs(self.pconv_classes, config=config)(outputs_imgs,
                                                                                             outputs_masks)
            for clazz in self.pconv_classes:
                img, mask = outputs_imgs[clazz], outputs_masks[clazz]
                assert img.dtype == torch.float32
                assert mask.dtype == torch.float32

    def test_input_shape(self):
        config = next(iter(self.realistic_config()))
        # We have to call each class distinctively
        pconv_calls = [clazz(**config.dict).to(self.device) for clazz in self.pconv_classes]

        # Good dtypes
        image = torch.rand(10, 3, 256, 256, dtype=torch.float32).to(self.device)
        mask = (torch.rand(10, 256, 256) > 0.5).to(torch.float32).to(self.device)
        try:
            for pconv_call in pconv_calls:
                pconv_call(image, mask)
        except TypeError as e:
            self.fail(str(e))

        image = (torch.rand(10, 256, 256) * 255).to(torch.float32).to(self.device)  # Bad shape, channels missing
        mask = (torch.rand(10, 256, 256) > 0.5).to(torch.float32).to(self.device)
        for pconv_call in pconv_calls:
            self.assertRaises(TypeError, pconv_call, image, mask)

        image = torch.rand(10, 3, 256, 256).to(torch.float32).to(self.device)
        mask = (torch.rand(10, 3, 256, 256) > 0.5).to(torch.float32).to(self.device)  # Bad shape, channels present
        for pconv_call in pconv_calls:
            self.assertRaises(TypeError, pconv_call, image, mask)

    def test_input_dtype(self):
        config = next(iter(self.realistic_config()))
        # We have to call each class distinctively
        pconv_calls = [clazz(**config.dict).to(self.device) for clazz in self.pconv_classes]

        # Good dtypes
        image = torch.rand(10, 3, 256, 256, dtype=torch.float32).to(self.device)
        mask = (torch.rand(10, 256, 256) > 0.5).to(torch.float32).to(self.device)
        try:
            for pconv_call in pconv_calls:
                pconv_call(image, mask)
        except TypeError as e:
            self.fail(str(e))

        image = (torch.rand(10, 3, 256, 256) * 255).to(torch.uint8).to(self.device)  # Bad dtype
        mask = (torch.rand(10, 256, 256) > 0.5).to(torch.float32).to(self.device)
        for pconv_call in pconv_calls:
            self.assertRaises(TypeError, pconv_call, image, mask)

        image = (torch.rand(10, 3, 256, 256) * 255).to(torch.float32).to(self.device)
        mask = (torch.rand(10, 256, 256) > 0.5).to(self.device)  # Bad Dtype
        for pconv_call in pconv_calls:
            self.assertRaises(TypeError, pconv_call, image, mask)

    def test_mask_values_binary(self):
        """The mask is a float tensor because the convolution doesn't operate on boolean tensors, however,
        its values are still 0.0 (False) OR 1.0 (True). The masks should NEVER have 0.34 or anything in
        between those two values.

        Technical explanation for why:
        masks are passed to the convolution with ones kernel, at that point, their values can be any integer
        since the convolution will sum ones together, so no float value can be created here.
        Then, we run torch.clip(mask, 0, 1). At this point, any integer value >= 1 becomes 1, leaving only 0 and 1s.
        Rince and repeat at next iteration."""
        image, mask = self.realistic_input()
        configs = self.realistic_config()
        outputs_imgs, outputs_masks = image, mask
        for config in configs:
            outputs_imgs, outputs_masks = self.run_pconvs(self.pconv_classes, config=config)(outputs_imgs,
                                                                                             outputs_masks)
            for mask in outputs_masks.values():
                assert ((mask == 1.0) | (
                    mask == 0.0)).all(), "All mask values should remain either 1.0 or 0.0, nothing in between."

    def test_dilation(self):
        image, mask = self.realistic_input()
        configs = self.realistic_config()
        # Enable bias on every PConv
        for i, c in enumerate(configs):
            c.dilation = max(1, i % 4)

        outputs_imgs, outputs_masks = image, mask
        for config in configs:
            outputs_imgs, outputs_masks = self.run_pconvs(self.pconv_classes, config=config)(outputs_imgs,
                                                                                             outputs_masks)
            self.compare(outputs_imgs, self.allclose)
            self.compare(outputs_masks, self.allclose)

    def test_bias(self):
        """This test is very sensitive to numerical errors.
        On my setup, this test passes when ran on GPU, but fails when ran on CPU. The most likely reason is that
        the CUDA backend's way to add the bias in the convolution differs from the Intel MKL way to add the bias,
        resulting in different numerical errors.

        Just inspect the min/mean/max values and see if they differ significantly, and if they don't then ignore this
        test failing, or send me a PR to fix it."""

        image, mask = self.realistic_input()
        configs = self.realistic_config()
        # Enable bias on every PConv
        for c in configs:
            c.bias = True

        outputs_imgs, outputs_masks = image, mask
        for config in configs:
            outputs_imgs, outputs_masks = self.run_pconvs(self.pconv_classes, config=config)(outputs_imgs,
                                                                                             outputs_masks)
            self.compare(outputs_imgs, self.allclose)
            self.compare(outputs_masks, self.allclose)

    def test_backpropagation(self):
        """Does a 3 step forward pass, and then attempts to backpropagate the resulting image
        to see if the gradient can be computed and wasn't lost along the way."""
        image, mask = self.realistic_input()
        configs = self.realistic_config()
        outputs_imgs, outputs_masks = image, mask
        for config in configs:
            outputs_imgs, outputs_masks = self.run_pconvs(self.pconv_classes, config=config)(outputs_imgs,
                                                                                             outputs_masks)

        for clazz in self.pconv_classes:
            try:
                outputs_imgs[clazz].sum().backward()
            except RuntimeError:
                self.fail(f"Could not compute the gradient for {clazz.__name__}")

    def test_memory_complexity(self):
        device = torch.device('cpu')
        image, mask = self.realistic_input(c=64, d=device)
        config = ConvConfig(64, 128, 9, stride=1, padding=3, bias=True)
        pconv_calls = [clazz(**config.dict).to(device) for clazz in self.pconv_classes]

        tolerance = 0.1  # 10 %
        max_mem_use = {
            PConvGuilin: 6_084_757_512,  # 5.67 GiB
            PConvRFR: 6_084_758_024,  # 5.67 GiB
            PConv2d: 2_405_797_640,  # 2.24 GiB
        }

        for pconv_call in pconv_calls:
            with profile(activities=[ProfilerActivity.CPU],
                         profile_memory=True, record_shapes=True, with_stack=True) as prof:
                # Don't forget to run grad computation as well, since that eats a lot of memory too
                out_im, _ = pconv_call(image, mask)
                out_im.sum().backward()

            # Stealing the total memory stat from the profiler
            total_mem = abs(
                list(filter(lambda fe: fe.key == "[memory]", list(prof.key_averages())))[0].cpu_memory_usage)

            # Printing how much mem used in total
            # print(f"{pconv_call.__class__.__name__} used {self.format_bytes(total_mem)} ({total_mem})")

            max_mem = (max_mem_use[pconv_call.__class__] * (1 + tolerance))
            assert total_mem < max_mem, f"{pconv_call.__class__.__name__} used {self.format_bytes(total_mem)}" \
                                        f" which is more than {self.format_bytes(max_mem)}"

    def test_iterated_equality(self):
        """
        Tests that even when iterating:
            1- The output images have the same values (do not diverge due to error accumulation for example)
            2- The output masks have the same values
            3- The outputted masks are just repeated along the channel dimension
        """
        image, mask = self.realistic_input()
        configs = self.realistic_config()

        outputs_imgs, outputs_masks = image, mask
        for config in configs:
            outputs_imgs, outputs_masks = self.run_pconvs(self.pconv_classes, config=config)(outputs_imgs,
                                                                                             outputs_masks)

            self.compare(outputs_imgs, self.allclose)
            self.compare(outputs_masks, self.allclose)

    def test_equality(self):
        config = ConvConfig(in_channels=3, out_channels=64, kernel_size=5)
        image, mask = self.mkinput(b=16, h=256, c=config.in_channels)

        outputs_imgs, outputs_masks = self.run_pconvs(self.pconv_classes, config)(
            image, mask)

        self.compare(outputs_imgs, self.allclose)

        self.compare(outputs_masks, self.allclose)

    @classmethod
    def realistic_input(cls, b=16, c=3, h=256, d=None) -> Tuple[Tensor, Tensor]:
        # 16 images, each of 3 channels and of height/width 256 pixels
        return cls.mkinput(b=b, c=c, h=h, d=cls.device if d is None else d)

    @classmethod
    def realistic_config(cls) -> Iterable[ConvConfig]:
        # These are the partial convs used in https://github.com/jingyuanli001/RFR-Inpainting
        # All have bias=False because in practice they're always followed by a BatchNorm2d anyway
        return (
            ConvConfig(3, 64, 7, stride=2, padding=3, bias=False),
            ConvConfig(64, 64, 7, stride=1, padding=3, bias=False),
            ConvConfig(64, 64, 7, stride=1, padding=3, bias=False),
            ConvConfig(64, 64, 7, stride=1, padding=3, bias=False),
            ConvConfig(64, 32, 3, stride=1, padding=1, bias=False),
        )

    @classmethod
    def mkinput(cls, b, c, h, d=None) -> Tuple[Tensor, Tensor]:
        if d is None:
            d = cls.device
        image = torch.rand(b, c, h, h).float().to(d)
        mask = (torch.rand(b, h, h) > 0.5).float().to(d)
        return image, mask

    @staticmethod
    def compare(values: Dict[Type[PConvLike], Tensor],
                comparator: Callable[[Tensor, Tensor], bool]):
        for (clazz1, out1), (clazz2, out2) in itertools.combinations(values.items(), 2):
            eq = comparator(out1, out2)
            if not eq:
                pshape(out1, out2, heading=True)
            assert eq, f"{clazz1.__name__ if hasattr(clazz1, '__name__') else 'class1'}'s doesn't match {clazz2.__name__ if hasattr(clazz2, '__name__') else 'class2'}'s output"

    @classmethod
    def run_pconvs(cls, pconvs: List[Type[PConvLike]], config: ConvConfig) -> Callable[
        [Union[Dict[Type[PConvLike], Tensor], Tensor],
         Union[Dict[Type[PConvLike], Tensor], Tensor]], Tuple[
            Dict[Type[PConvLike], Tensor], Dict[Type[PConvLike], Tensor]]]:
        """Returns a closure that :
        Initialise each PConvLike class with the provided config,
        set their weights and biases to be equal, and run each of them onto the
        input(s) images/masks. Then saves the output in a dict that match the class to
        the output. Returns that dict.
        The closure can be called with either a specific input per class, or one input
        which will be shared among every class.

        This method's signature is admittedly a bit unwieldy...

        :param pconvs: the list of PConvLike classes to run
        :param config: the ConvConfig to use for those classes
        :return: The returned closure takes either two tensors, or two dict of tensors
        where keys are the corresponding PConv classes which to call it on
        """

        def inner(imgs: Union[Dict[Type[PConvLike], Tensor], Tensor],
                  masks: Union[Dict[Type[PConvLike], Tensor], Tensor]) -> \
            Tuple[
                Dict[Type[PConvLike], Tensor], Dict[Type[PConvLike], Tensor]]:
            if not isinstance(imgs, dict):
                imgs = {clazz: imgs for clazz in pconvs}
            if not isinstance(masks, dict):
                masks = {clazz: masks for clazz in pconvs}
            outputs_imgs = dict()
            outputs_masks = dict()
            w = None
            b = None
            for clazz in pconvs:
                # noinspection PyArgumentList
                pconv = clazz(**config.dict).to(cls.device)
                if config.bias:
                    if b is None:
                        b = pconv.get_bias()
                    else:
                        pconv.set_bias(b.clone())

                if w is None:
                    w = pconv.get_weight()
                else:
                    pconv.set_weight(w.clone())

                out_img, out_mask = pconv(imgs[clazz].clone(), masks[clazz].clone())
                outputs_imgs[clazz] = out_img
                outputs_masks[clazz] = out_mask
            return outputs_imgs, outputs_masks

        return inner

    @classmethod
    def channelwise_allclose(cls, x):
        close = True
        for channel1, channel2 in itertools.combinations(x.transpose(0, 1), 2):
            close &= cls.allclose(channel1, channel2)
        return close

    @classmethod
    def channelwise_almost_eq(cls, x):
        close = True
        for channel1, channel2 in itertools.combinations(x.transpose(0, 1), 2):
            close &= cls.almost_eq(channel1, channel2)
        return close

    @staticmethod
    def almost_eq(x, y):
        return torch.allclose(x, y, rtol=0, atol=2e-3)

    @staticmethod
    def allclose(x, y):
        return torch.allclose(x, y, rtol=1e-5, atol=1e-8)

    @staticmethod
    def format_bytes(size):
        # 2**10 = 1024
        power = 2 ** 10
        n = 0
        power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
        while abs(size) > power:
            size /= power
            n += 1
        suffix = power_labels[n] + 'iB'
        return f"{size:.2f} {suffix}"

    if __name__ == "__main__":
        unittest.main()
