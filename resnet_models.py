import torch
from torch.nn import Module, Sequential, Conv3d, BatchNorm3d, ReLU, MaxPool3d, AvgPool3d, Linear, Dropout

class MainPath(Module):
    def __init__(self, in_channels, filters, kernel_size, stride=1):
        super().__init__()
        F1, F2, F3 = filters
        self.main_path = Sequential(
            Conv3d(in_channels, F1, kernel_size=1, stride=stride),
            BatchNorm3d(F1),
            ReLU(),
            Conv3d(F1, F2, kernel_size=kernel_size, padding=kernel_size//2),
            BatchNorm3d(F2),
            ReLU(),
            Conv3d(F2, F3, kernel_size=1),
            BatchNorm3d(F3),
        )
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv3d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        y = self.main_path(x)
        return y

class IdentityBlock(MainPath):
    def __init__(self, in_channels, filters, kernel_size):
        super().__init__(in_channels, filters, kernel_size)
        self.relu = ReLU()

    def forward(self, x):
        y = self.relu(self.main_path(x) + x)
        return y
    
class ConvolutionalBlock(MainPath):
    def __init__(self, in_channels, filters, kernel_size):
        super().__init__(in_channels, filters, kernel_size, stride=2)
        self.relu = ReLU()
        self.shortcut_path = Sequential(
            Conv3d(in_channels, filters[2], kernel_size=1, stride=2),
            BatchNorm3d(filters[2])
        )
        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv3d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        y = self.relu(self.main_path(x) + self.shortcut_path(x))
        return y
    
class ResNet3D(Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = Sequential(
            Conv3d(1, 16, kernel_size=7, stride=2, padding=3),  # Halved channels: 32 -> 16
            BatchNorm3d(16),
            MaxPool3d(kernel_size=3, stride=2, padding=1),
            ConvolutionalBlock(16, [16, 16, 64], kernel_size=3),  # Halved: 32 -> 16, 128 -> 64
            Dropout(0.2),
            IdentityBlock(64, [16, 16, 64], kernel_size=3),
            IdentityBlock(64, [16, 16, 64], kernel_size=3),
            ConvolutionalBlock(64, [32, 32, 128], kernel_size=3),  # Halved: 64 -> 32, 256 -> 128
            Dropout(0.3),
            IdentityBlock(128, [32, 32, 128], kernel_size=3),
            IdentityBlock(128, [32, 32, 128], kernel_size=3),
            IdentityBlock(128, [32, 32, 128], kernel_size=3),
            ConvolutionalBlock(128, [64, 64, 256], kernel_size=3),  # Halved: 128 -> 64, 512 -> 256
            Dropout(0.3),
            IdentityBlock(256, [64, 64, 256], kernel_size=3),
            IdentityBlock(256, [64, 64, 256], kernel_size=3),
            IdentityBlock(256, [64, 64, 256], kernel_size=3),
            IdentityBlock(256, [64, 64, 256], kernel_size=3),
            IdentityBlock(256, [64, 64, 256], kernel_size=3),
            ConvolutionalBlock(256, [128, 128, 512], kernel_size=3),  # Halved: 256 -> 128, 2048 -> 512
            Dropout(0.3),
            IdentityBlock(512, [128, 128, 512], kernel_size=3),
            IdentityBlock(512, [128, 128, 512], kernel_size=3),
            AvgPool3d(kernel_size=1, stride=1)
        )
        self.classification_layer = Linear(512, num_classes)  # Adjusted to match halved final layer
        # self.apply(self._init_weights)

    def forward(self, x):
        y = self.network(x).reshape((x.shape[0], -1))
        y = self.classification_layer(y)
        return y

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, torch.nn.Conv3d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
