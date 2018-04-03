from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import pandas as pd
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import skimage.transform
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
from pathlib import Path
import cv2


class UISDSC(data.Dataset):
    """`UISDSC <https://dscomp.ibest.uidaho.edu/data>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'https://dscomp.ibest.uidaho.edu/uploads/train.csv',
        'https://dscomp.ibest.uidaho.edu/uploads/train_labels.csv',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            if Path(self.root, self.raw_folder, filename).exists():
                continue
            file_path = os.path.join(self.root, self.raw_folder, filename)
            print("File: {}".format(file_path))
            with open(file_path, 'wb') as f:
                f.write(data.read())

        # process and save as torch files
        print('Processing...')

        labelpath = Path('.').joinpath(self.root, self.raw_folder, 'train_labels.csv')
        imagepath = Path('.').joinpath(self.root, self.raw_folder, 'train.csv')
        train_labels, test_labels, train_images, test_images = transform_images(labelpath, imagepath)

        training_set = (
            train_images, train_labels
        )
        test_set = (
            test_images, test_labels
        )

        print("Saving cached images...")
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def rotation(img, rotation_angle=15):
    rows, cols = img.shape
    angle = np.random.randint(-rotation_angle, rotation_angle)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img = cv2.warpAffine(img, M, (cols, rows))
    return img


def shift_pixels(img, shift_range=1):
    rows, cols = img.shape
    shift = np.random.randint(-shift_range, shift_range)
    M = np.float32([[1, 0, shift], [0, 1, shift]])
    return cv2.warpAffine(img, M, (cols, rows))


def add_noise(img, gauss_var=0.02):
    img = skimage.util.random_noise(img, mode='gaussian', var=gauss_var)
    return img


def transform_images(labelpath, imagepath):
    # Use a train test split function
    labels = pd.read_csv(labelpath).values
    images = pd.read_csv(imagepath).values
    labels = labels[:, 1]
    images = images[:, 1:].reshape(images.shape[0], 24, 24)
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2)
    train_image_list = []
    train_label_list = []
    start = timer()
    for i, (im, l) in enumerate(zip(train_images, train_labels)):
        for k in range(20):
            train_image_list.append(add_noise(shift_pixels(rotation(im))))
            train_label_list.append(l)
        if i % 1000 == 0:
            print("{:08d}: {:.6f}".format(i, timer() - start))
            start = timer()
    for i, im in enumerate(test_images):
        test_images[i] = add_noise(im)
    train_image_list = np.asarray(train_image_list).reshape(len(train_image_list), 1, 24, 24)
    train_label_list = np.asarray(train_label_list)
    test_labels = np.asarray(test_labels)
    test_images = np.asarray(test_images).reshape(len(test_images), 1, 24, 24)

    return train_label_list, test_labels, torch.from_numpy(train_image_list), torch.from_numpy(test_images)
