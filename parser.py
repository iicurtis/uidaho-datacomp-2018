import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from loader import TestLoader
from pathlib import Path

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                    help='input batch size for testing (default: 10000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--load', type=str, default=None,
                    help='directory to load model from')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

test_loader = torch.utils.data.DataLoader(
    TestLoader('sub_test.csv', transform=transforms.Compose([
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 13)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


model = Net()
savedir = Path("save")
save = sorted(list(savedir.glob("*.pth")))[-1]
print("Loading file: {}".format(save))
state_dict = torch.load(save)
model.load_state_dict(state_dict)

if args.cuda:
    model.cuda()


def test():
    print("Running")
    model.eval()
    lsize = len(test_loader) * args.test_batch_size
    results = np.ones((lsize, 6), dtype=np.int64)
    for data_list, index in test_loader:
        for k, data in enumerate(data_list):
            if args.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            output = model(data)
            if k in [0, 2, 4]:
                pred = output[:, :10].data.max(1, keepdim=True)[1] # get the index of the max log-probability
            else:
                pred = output[:, 10:].data.max(1, keepdim=True)[1] + 10
            results[index, k] = np.squeeze(pred.cpu().numpy())

    print("Evaluating")
    for k in results:
        if k[1] == 12:
            if k[3] == 10:
                if k[0] == k[2] + k[4]:
                    k[5] = 0
            elif k[3] == 11:
                if k[0] == k[2] - k[4]:
                    k[5] = 0
        elif k[3] == 12:
            if k[1] == 10:
                if k[4] == k[0] + k[2]:
                    k[5] = 0
            elif k[1] == 11:
                if k[4] == k[0] - k[2]:
                    k[5] = 0
        #unique, counts = np.unique(k, return_counts=True)
        if np.count_nonzero(k == 12) != 1:
            k[5] = 3
        if (np.count_nonzero(k == 10) + np.count_nonzero(k == 11)) != 1:
            k[5] = 2
    df = pd.DataFrame(results)
    df.replace(12, "=")
    df.replace(11, "-")
    df.replace(10, "+")
    df.to_csv("curtis_debug.csv")
    sub = pd.DataFrame({"label": df[5]})
    sub.to_csv("curtis_submission.csv", index_label="index")
    print("Done HAHAHAHAHAHAHA")


test()
