import torch
import torch.nn as nn
from typing import Tuple
import numpy as np


class ConvBlock(nn.Module):
    def __init__(self, in_size: int, out_size: int,
                 kernel_size: Tuple[int, int]):
        super().__init__()
        hidden_size = out_size // 4
        self.conv1 = nn.Conv2d(in_size, hidden_size, kernel_size=kernel_size,
                               padding=(kernel_size[0]//2, 1))
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_size, out_size, kernel_size=kernel_size,
                               padding=(kernel_size[0]//2, 1))
        self.bn2 = nn.BatchNorm2d(out_size)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        y = self.relu2(h)
        return y


class ConvModel(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int],
        kernel_size: Tuple[int, int],
        layer_depth: int,
        out_features: int,
    ):
        super(ConvModel, self).__init__()
        self.input_shape = input_shape
        self.layer_depth = layer_depth
        self.conv_layers = nn.ModuleList()
        self.flatten = nn.Flatten()

        channels, last_conv_output_shape = self._calc_layer_shape()
        for i in range(len(channels)-1):
            self.conv_layers.append(ConvBlock(*channels[i: i+2], kernel_size=kernel_size))
            self.conv_layers.append(nn.MaxPool2d(kernel_size=(1, 2)))

        self.fn = nn.Linear(np.prod(last_conv_output_shape), out_features)

    def _calc_layer_shape(self):
        channels = [1]
        _shape = self.input_shape
        for i in range(1, self.layer_depth+1):
            channels.append(8*2**(i+1))
            _shape = (_shape[0], _shape[1]//2)
        last_conv_output_shape = channels[-1] * _shape[0] * _shape[1]
        return channels, last_conv_output_shape

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        x = self.flatten(x)
        out = self.fn(x)
        return out


if __name__ == '__main__':
    model = ConvModel((12, 400), (3, 3), 3, 10)
    print(model)
    print(model(torch.randn(1, 1, 12, 400)))
