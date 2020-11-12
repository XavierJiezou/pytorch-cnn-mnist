import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            # [BATCH_SIZE, 1, 28, 28]
            nn.Conv2d(1, 32, 5, 1, 2),
            # [BATCH_SIZE, 32, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),
            # [BATCH_SIZE, 32, 14, 14]
            nn.Conv2d(32, 64, 5, 1, 2),
            # [BATCH_SIZE, 64, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2)
            # [BATCH_SIZE, 64, 7, 7]
        )
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        y = self.fc(x)
        return y
