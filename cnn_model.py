import torch.nn as nn
import torch.nn.functional as F


class Simple3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Simple3DCNN, self).__init__()
        # in_channels = 1 as the voxel grids only have 1 channel
        self.conv1 = nn.Conv3d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv3d(
            in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv3d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.2)  # Adding dropout for regularization
        self.fc1 = nn.Linear(64 * 4 * 4 * 4, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4 * 4)  # Flatten before fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))  # Applying dropout before the last layer
        x = self.fc2(x)
        return x