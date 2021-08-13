from dataclasses import dataclass, asdict


@dataclass
class ConvConfig:
    in_channels: int
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    dilation: int = 1
    bias: bool = False

    def copy(self):
        # noinspection PyArgumentList
        return self.__class__(**asdict(self))

    @property
    def dict(self):
        return asdict(self)
