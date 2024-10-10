import torch.nn as nn
import torch.nn.functional as F


class Simple3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 4 * 4 * 4, 128)  # Adjust dimensions based on the output size after convolutions
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4 * 4)  # Flatten before fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Simple3DCNN_BN(nn.Module):
    def __init__(self, num_classes):
        super(Simple3DCNN_BN, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(16)  # Batch normalization after conv1
        
        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(32)  # Batch normalization after conv2
        
        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(64)  # Batch normalization after conv3
        
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 4 * 4 * 4, 128)  # Adjust dimensions based on the output size after convolutions
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Apply batch normalization and ReLU
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Apply batch normalization and ReLU
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Apply batch normalization and ReLU
        
        x = x.view(-1, 64 * 4 * 4 * 4)  # Flatten before fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
