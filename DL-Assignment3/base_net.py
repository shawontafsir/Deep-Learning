import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as transforms


class BaseNet(nn.Module):

    def __init__(self, layers_count):
        super().__init__()

        self.layers_count = layers_count
        self.neurons_count = 10

        # Parameters for the first layer
        self.input_channels_count = 3
        self.filters_count = 16
        self.filter_size = 3
        self.stride = 1
        self.padding = 1

        # Define the first layer
        self.first_convolution = nn.Conv2d(
            self.input_channels_count,
            self.filters_count,
            kernel_size=self.filter_size,
            stride=self.stride,
            padding=self.padding
        )
        self.first_batch_normalization = nn.BatchNorm2d(self.filters_count)
        self.relu = nn.ReLU(inplace=True)

        # Create and get modules of convolution layers
        self.module1 = self.create_module(self.filters_count, 16, 3, stride=1)
        self.module2 = self.create_module(16, 32, 3)
        self.module3 = self.create_module(32, 64, 3)

        # Final layer:
        # 1. Apply global average pooling, 2. a dense fully connected softmax layer
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))

        # defining fully connected layer where in_features is 64 received from self.module3
        self.fc_layer = nn.Linear(64, self.neurons_count)
        self.softmax_layer = nn.Softmax(dim=1)

    def create_module(self, input_channels_count, output_channels_count, filter_size, stride=None):
        layers = list()
        for layer_ind in range(self.layers_count):
            # stride will depend on layer if not given in cases like 2nd & 3rd modules
            if stride is None:
                stride = 2 if layer_ind == 0 else 1

            layer_convolution = nn.Conv2d(
                    input_channels_count,
                    output_channels_count,
                    kernel_size=filter_size,
                    stride=stride,
                    padding=1
                )
            layer_batch_norm = nn.BatchNorm2d(output_channels_count)
            layer_relu = nn.ReLU(inplace=True)

            layers.append(layer_convolution)
            layers.append(layer_batch_norm)
            layers.append(layer_relu)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first_convolution(x)
        x = self.first_batch_normalization(x)
        x = self.relu(x)
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.global_average_pool(x)
        x = self.fc_layer(x)
        x = self.softmax_layer(x)

        return x
