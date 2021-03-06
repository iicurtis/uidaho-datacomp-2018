import argparse
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
import skimage

import models

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
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


def add_noise(img, gauss_var=0.02):
    img = skimage.util.random_noise(img, mode='gaussian', var=gauss_var)
    return img


class TestLoader(Dataset):
    """UIdaho CSV dataset."""

    def __init__(self, test_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train_data = pd.read_csv(test_file).values
        self.train_data = self.train_data[:, 1:].reshape(self.train_data.shape[0], 1, 24, 120)
        self.transform = transform

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        img = self.train_data[idx]
        # img = img.astype(np.uint8)
        #  img = add_noise(img)

        images = []
        for k in range(5):
            bgn = k*24
            end = (k+1)*24
            imgchar = img[:, :, bgn:end]
            imgchar = torch.from_numpy(imgchar).float()
            if self.transform is not None:
                imgchar = self.transform(imgchar)
            images.append(imgchar)

        return images, idx


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

test_loader = torch.utils.data.DataLoader(
    TestLoader('data/raw/sub_test.csv',
        #        transform=transforms.Compose([
        #            # transforms.Resize(28),
        # transforms.Normalize((0.1307,), (0.3081,)),
        #        ])),
               ),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


model = models.Vgg(nchannels=1, nfilters=8, nclasses=13)
if args.cuda:
    model = torch.nn.DataParallel(model, device_ids=list(range(1)))
    model = model.cuda()
# savedir = Path("./results/2018-04-11_10-50-51")
savedir = sorted(list(Path("./results/").iterdir()))[-1]
save = sorted(list(savedir.glob("**/*.pth")))[-1]
print("Loading file: {}".format(save))
state_dict = torch.load(save)
model.load_state_dict(state_dict)

if args.cuda:
    model.cuda()


def test():
    print("Running")
    model.eval()
    lsize = len(test_loader) * args.test_batch_size
    results = np.zeros((lsize, 6), dtype=np.int64)
    badres = np.zeros((lsize, 5), dtype=np.float64)
    for data_list, index in test_loader:
        for k, data in enumerate(data_list):
            if args.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            output = model(data)
            if k in [0, 2, 4]:
                pred = output[:, :10].data
                predtop = pred.topk(2, dim=1)  # get the index of the max log-probability
                diff = predtop[0][:, 0] - predtop[0][:, 1]
                pred = pred.max(1, keepdim=True)[1]
            else:
                pred = output[:, 10:].data.max(1, keepdim=True)[1] + 10
            badres[index, k] = np.squeeze(diff.cpu().numpy())
            results[index, k] = np.squeeze(pred.cpu().numpy())

    print("Evaluating")
    for k in results:
        if k[1] == 12:
            if k[3] == 10:
                if k[0] == k[2] + k[4]:
                    k[5] = 1
            elif k[3] == 11:
                if k[0] == k[2] - k[4]:
                    k[5] = 1
        elif k[3] == 12:
            if k[1] == 10:
                if k[4] == k[0] + k[2]:
                    k[5] = 1
            elif k[1] == 11:
                if k[4] == k[0] - k[2]:
                    k[5] = 1
        # unique, counts = np.unique(k, return_counts=True)
        if np.count_nonzero(k == 12) != 1:
            k[5] = 3
            print("WARN: {} has more than one equals!!".format(k[0]))
        if (np.count_nonzero(k == 10) + np.count_nonzero(k == 11)) != 1:
            k[5] = 2
            print("WARN: {} has more than one operator!!".format(k[0]))
    df = pd.DataFrame(results)
    df.to_csv("curtis_debug.csv")
    sub = pd.DataFrame({"label": df[5]})
    sub.to_csv("curtis_submission.csv", index_label="index")
    print("Done HAHAHAHAHAHAHA")
    questionable = np.argwhere(badres < 4)
    for q in questionable:
        r = results[q[0]]
        print("{:06d}:{}  {} {} {} {} {} | {}  {}".format(q[0], q[1], r[0], r[1],
                                                      r[2], r[3], r[4], r[5], badres[q[0], q[1]]))


test()
