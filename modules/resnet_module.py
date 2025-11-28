from torch import nn
from torch import Tensor
from compressai.layers import conv3x3


class ResidualBottleneckBlock(nn.Module):
    """
    Optimized Residual Bottleneck Block.

    Includes Batch Normalization and proper ReLU placement as per standard
    ResNet architecture (He2016).
    """

    def __init__(self, in_ch: int, out_ch: int, expansion: int = 4):
        super().__init__()

        mid_ch = out_ch // expansion

        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(mid_ch)

        self.conv2 = conv3x3(mid_ch, mid_ch)
        self.bn2 = nn.BatchNorm2d(mid_ch)

        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResidualBottleneckBlockWithStride(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_layers: int = 3):
        super().__init__()

        self.conv_down = nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=2, padding=2)

        layers = []
        for _ in range(num_layers):
            layers.append(ResidualBottleneckBlock(out_ch, out_ch))

        self.res_blocks = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv_down(x)
        out = self.res_blocks(out)
        return out


class ResidualBottleneckBlockWithUpsample(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_layers: int = 3):
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(ResidualBottleneckBlock(in_ch, in_ch))
        self.res_blocks = nn.Sequential(*layers)

        self.conv_up = nn.ConvTranspose2d(
            in_ch, out_ch, kernel_size=5, stride=2, padding=2, output_padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.res_blocks(x)
        out = self.conv_up(out)
        return out
