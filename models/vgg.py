# lenet.py

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Vgg']


class Vgg(nn.Module):
    def __init__(self, nchannels=3, nclasses=10, nfilters=8):
        super(Vgg, self).__init__()
        self.nfilters = nfilters
        self.classifier = nn.Sequential(
            nn.Linear(nfilters * 64, nfilters * 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(nfilters * 64, nclasses)
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(nchannels, nfilters * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(nfilters * 8),
            nn.ReLU(True),
            nn.Conv2d(nfilters * 8, nfilters * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(nfilters * 8),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(nfilters * 8, nfilters * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(nfilters * 16),
            nn.ReLU(True),
            nn.Conv2d(nfilters * 16, nfilters * 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(nfilters * 16),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(nfilters * 16, nfilters * 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(nfilters * 32),
            nn.ReLU(True),
            nn.Conv2d(nfilters * 32, nfilters * 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(nfilters * 32),
            nn.ReLU(True),
            nn.Conv2d(nfilters * 32, nfilters * 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(nfilters * 32),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(nfilters * 32, nfilters * 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(nfilters * 64),
            nn.ReLU(True),
            nn.Conv2d(nfilters * 64, nfilters * 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(nfilters * 64),
            nn.ReLU(True),
            nn.Conv2d(nfilters * 64, nfilters * 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(nfilters * 64),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)
