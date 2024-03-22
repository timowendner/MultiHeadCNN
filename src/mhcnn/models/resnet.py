import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, channels: int, block_size: int, kernel_size: int = 3, dropout: float = 0.0):
        super(ResBlock, self).__init__()
        layers = []
        for i in range(block_size):
            layers.extend([
                nn.Conv2d(
                    channels, channels, kernel_size, padding=kernel_size//2
                ),
                nn.BatchNorm2d(channels),
                nn.Dropout2d(p=dropout),
                nn.ReLU(),
            ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x) + x
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        conv_layer: list[int],
        linear_layer: list[int],
        input_shape: list[int],
        classes: int,
        kernel: int = 3,
        block_size: int = 2,
        block_count: int = 2,
        dropout: float = 0.2,
        padding: int = None,
    ):
        super(ResNet, self).__init__()
        conv = []
        last = input_shape[0]
        for current in conv_layer:
            conv.extend([
                nn.Conv2d(last, current, kernel, padding=padding),
                nn.ReLU()
            ])
            for _ in range(block_count):
                conv.append(
                    ResBlock(current, block_size, kernel, dropout)
                )
            conv.append(nn.MaxPool2d(2))
            last = current
        conv = conv[:-1]
        self.conv = nn.Sequential(*conv)

        linear = []
        last = self._get_conv_out_size(input_shape)
        for current in linear_layer:
            linear.append(nn.Linear(last, current))
            linear.append(nn.ReLU())
            last = current
        self.linear = nn.Sequential(*linear, nn.Linear(last, classes))

    def _get_conv_out_size(self, shape):
        data = torch.zeros((1, *shape))
        data = self.conv(data)
        data = data.flatten(1)
        size = int(data.shape[1])
        return size

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.linear(x)
        return x
