import torch.nn as nn
import torch.nn.functional as F


__all__ = ['Uids']


class Uids(nn.Module):
    def __init__(self):
        super(Uids, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.conv1_2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.conv2_1 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv2_drop = nn.Dropout2d(p=0.2)
        self.conv3_drop = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(4608, 2048)
        self.fc2 = nn.Linear(2048, 13)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.max_pool2d(F.relu(self.conv1_2(x)), 2)
        x = F.relu(self.conv2_1(x))
        x = self.conv2_drop(F.max_pool2d(F.relu(self.conv2_2(x)), 2))
        x = self.conv3_drop(F.relu(self.conv3_1(x)))
        x = F.max_pool2d(F.relu(self.conv3_2(x)), 2)
        x = x.view(-1, 4608)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
