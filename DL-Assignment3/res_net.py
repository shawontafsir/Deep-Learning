from torch import nn


class ResidualBlock(nn.Module):

    def __init__(self, input_channels_count, output_channels_count):
        super().__init__()

        # Parameters for the first layer
        self.filters_count = output_channels_count
        self.filter_size = 3
        self.stride = 1
        self.padding = 1

        # He initialization
        self.apply(self.init_weights)

        # Define skip connection
        self.skip_convolution = nn.Conv2d(
            input_channels_count,
            self.filters_count,
            kernel_size=1,
            stride=self.stride,
            padding=0
        )
        self.skip_batch_normalization = nn.BatchNorm2d(self.filters_count)

        # Define the first layer
        self.first_convolution = nn.Conv2d(
            input_channels_count,
            self.filters_count,
            kernel_size=self.filter_size,
            stride=self.stride,
            padding=self.padding
        )
        self.first_batch_normalization = nn.BatchNorm2d(self.filters_count)
        self.relu = nn.ReLU(inplace=True)

        # Define the second convolution layer
        self.second_convolution = nn.Conv2d(
            self.filters_count,
            self.filters_count,
            kernel_size=self.filter_size,
            stride=self.stride,
            padding=self.padding
        )
        self.second_batch_normalization = nn.BatchNorm2d(self.filters_count)

    @staticmethod
    def init_weights(obj):
        if isinstance(obj, nn.Linear) or isinstance(obj, nn.Conv2d):
            nn.init.kaiming_uniform_(obj.weight)

    def forward(self, x):
        identity = self.skip_batch_normalization(self.skip_convolution(x))
        x = self.first_convolution(x)
        x = self.first_batch_normalization(x)
        x = self.relu(x)
        x = self.second_convolution(x)
        x = self.second_batch_normalization(x)
        x = x + identity
        x = self.relu(x)

        return x


class ResNet(nn.Module):

    def __init__(self, residual_blocks_count):
        super().__init__()

        self.residual_blocks_count = residual_blocks_count
        self.neurons_count = 10

        # Parameters for the first layer
        self.input_channels_count = 3
        self.filters_count = 16
        self.filter_size = 3
        self.stride = 1
        self.padding = 1

        # He initialization
        self.apply(self.init_weights)

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

        # Create and get modules of residual blocks
        self.module1 = self.create_module(self.filters_count, 16)
        self.module2 = self.create_module(16, 32)
        self.module3 = self.create_module(32, 64)

        # Final layer:
        # 1. Apply global average pooling, 2. a dense fully connected softmax layer
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Defining fully connected layer where in_features count is 64 received from self.module3
        self.fully_connected_layer = nn.Linear(64, self.neurons_count)
        self.softmax_layer = nn.Softmax(dim=1)

    @staticmethod
    def init_weights(obj):
        if isinstance(obj, nn.Linear) or isinstance(obj, nn.Conv2d):
            nn.init.kaiming_uniform_(obj.weight)

    def create_module(self, input_channels_count, output_channels_count):
        blocks = list()

        for _ in range(self.residual_blocks_count):
            block = ResidualBlock(input_channels_count, output_channels_count)
            blocks.append(block)

            # From 2nd to each block, both the number of input and output channels are same
            input_channels_count = output_channels_count

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.first_convolution(x)
        x = self.first_batch_normalization(x)
        x = self.relu(x)
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.global_average_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fully_connected_layer(x)
        x = self.softmax_layer(x)

        return x
