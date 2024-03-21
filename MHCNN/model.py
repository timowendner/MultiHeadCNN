import torch
from torch import nn
import numpy as np
from torch import sin, cos, pow, Tensor
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, layers: list[int]) -> None:
        super(CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_out_size = self._get_conv_out_size()
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
            nn.Sigmoid(),
        )

    def _get_conv_out_size(self):
        sample_input = torch.zeros(
            (1, 3, 32, 32), dtype=torch.float32)
        conv_out = self.conv_layers(sample_input)
        return int(conv_out.view(-1).size(0))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, self.conv_out_size)
        x = self.fc_layers(x)
        return x
