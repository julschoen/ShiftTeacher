import torch
import torch.nn as nn
from torchvision.models import resnet18
from torch.nn import functional as F


class LeNetShiftTeacher(nn.Module):
    def __init__(self,params):
        super(LeNetShiftTeacher, self).__init__()
        width = params.filters
        self.convnet = nn.Sequential(
            nn.Conv2d(3, 3 * width, kernel_size=(5, 5)),
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

    def forward(self, x):
        batch_size = x.shape[0]
        features = self.convnet(x)

        features = features.mean(dim=[-1, -2])
        features = features.view(batch_size, -1)

        return self.fc_shift(features).squeeze


def save_hook(module, input, output):
    setattr(module, 'output', output)


class ResNetShiftTeacher(nn.Module):
    def __init__(self, params):
        super(ResNetShiftTeacher, self).__init__()
        self.features_extractor = resnet18(pretrained=False)
        self.features_extractor.conv1 = nn.Conv2d(
            3, 64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        nn.init.kaiming_normal_(self.features_extractor.conv1.weight,
                                mode='fan_out', nonlinearity='relu')

        self.features = self.features_extractor.avgpool
        self.features.register_forward_hook(save_hook)

        # half dimension as we expect the model to be symmetric
        self.shift_estimator = nn.Linear(512, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        batch_size = x.shape[0]
   
        self.features_extractor(x)
        features = self.features.output.view([batch_size, -1])

        return self.act(self.shift_estimator(features)).squeeze()
