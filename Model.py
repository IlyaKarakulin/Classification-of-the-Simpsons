import torch.nn as nn


class Model(nn.Module):
    def __init__(self, n_classes=42):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=7, stride=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=256, stride=1)
        )

    def forward(self, x):
        x = self.conv1(x)

        return x
    