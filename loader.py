import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import skimage


def crop(im, height, width):
    # im = Image.open(infile)
    imgwidth, imgheight = im.size
    for i in range(imgheight//height):
        for j in range(imgwidth//width):
            # print (i,j)
            box = (j*width, i*height, (j+1)*width, (i+1)*height)
            yield im.crop(box)


def add_noise(img, gauss_var=0.02):
    img = skimage.util.random_noise(img, mode='gaussian', var=gauss_var)
    return img


class CSVLoader(Dataset):
    """UIdaho CSV dataset."""

    def __init__(self, train_file, label_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train_data = pd.read_csv(train_file)
        self.train_labels = pd.read_csv(label_file)
        self.transform = transform

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        target = self.train_labels.iloc[idx, 1]
        img = self.train_data.iloc[idx, 1:].as_matrix().reshape(24, 24) * 255
        img = img.astype(np.uint8)
        img = Image.fromarray(img, mode='L')
        if self.transform is not None:
            img = self.transform(img)

        return img, target


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
        self.train_data = pd.read_csv(test_file)
        self.transform = transform

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        img = self.train_data.iloc[idx, 1:].as_matrix().reshape(24, 120)
        img = img.astype(np.uint8)
        img = add_noise(img) * 255
        img = Image.fromarray(img, mode='L')
        img.save("data/img/prob_{:08d}.png".format(idx))

        images = []
        for k, imgchar in enumerate(crop(img, 24, 24)):
            if self.transform is not None:
                imgchar = self.transform(imgchar)
            images.append(imgchar)

        return images, idx
