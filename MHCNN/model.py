import torch
from torch import nn
import numpy as np
from torch import sin, cos, pow, Tensor
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.0):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                channels, channels, kernel_size, padding=kernel_size//2
            ),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(p=dropout),
            nn.ReLU(),
            nn.Conv2d(
                channels, channels, kernel_size, padding=kernel_size//2
            ),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(p=dropout),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.layers(x) + x
        return out


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0):
        super(CNNBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, padding=padding,
            ),
            nn.ReLU(),
            nn.Conv2d(
                out_channels, out_channels, kernel_size, padding=padding,
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class CNN(nn.Module):
    def __init__(self, layers: list[int]) -> None:
        super(CNN, self).__init__()
        last = 3
        self.conv_layers = nn.ModuleList([])
        for i, channels in enumerate(layers):
            self.conv_layers.append(nn.Sequential(
                CNNBlock(last, channels, kernel_size=3, padding=1),
                ResBlock(channels, kernel_size=3, dropout=0.2),
                ResBlock(channels, kernel_size=3, dropout=0.2),
            ))
            if i != len(layers) - 1:
                self.conv_layers.append(nn.MaxPool2d(2))
            last = channels

        # self.conv_layers = nn.Sequential(
        #     CNNBlock(3, 32, kernel_size=3, padding=0),
        #     ResBlock(32, kernel_size=3, dropout=0.2),
        #     ResBlock(32, kernel_size=3, dropout=0.2),
        #     nn.MaxPool2d(2),
        #     CNNBlock(32, 64, kernel_size=3, padding=0),
        #     ResBlock(64, kernel_size=3, dropout=0.2),
        #     ResBlock(64, kernel_size=3, dropout=0.2),
        #     nn.MaxPool2d(2),
        #     CNNBlock(64, 64, kernel_size=3, padding=1),
        #     ResBlock(64, kernel_size=3, dropout=0.2),
        #     ResBlock(64, kernel_size=3, dropout=0.2),
        # )

        self.conv_out_size = self._get_conv_out_size()
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Softmax(dim=1)
        )

    def _get_conv_out_size(self):
        data = torch.zeros(
            (1, 3, 32, 32), dtype=torch.float32)
        for layer in self.conv_layers:
            data = layer(data)
        # conv_out = self.conv_layers(sample_input)
        size = int(data.view(-1).size(0))
        print(size)
        return size

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = x.view(-1, self.conv_out_size)
        x = self.fc_layers(x)
        return x
