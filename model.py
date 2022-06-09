import torch
import torch.nn as nn


class ShiftTeacher(nn.Module):
    def __init__(self, params):
        super(LeNetShiftPredictor, self).__init__()
        width = params.filters
        self.convnet = nn.Sequential(
            nn.Conv2d(1 * 2, 3 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(3 * width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(3 * width, 8 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(8 * width),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(8 * width, 60 * width, kernel_size=(5, 5)),
            nn.BatchNorm2d(60 * width),
            nn.ReLU()
        )

        self.fc_shift = nn.Sequential(
            nn.Linear(60 * width, 42 * width),
            nn.BatchNorm1d(42 * width),
            nn.ReLU(),
            nn.Linear(42 * width, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        features = self.convnet(torch.cat([x1, x2], dim=1))

        features = features.mean(dim=[-1, -2])
        features = features.view(batch_size, -1)
        shift = self.fc_shift(features)

        return shift.squeeze()