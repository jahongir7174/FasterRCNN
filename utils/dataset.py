import os
import random

import numpy
import torch
from PIL import Image
from torch.utils import data

from utils import util

FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'


class Dataset(data.Dataset):
    def __init__(self, filenames, input_size=640, augment=False):
        self.augment = augment
        self.input_size = input_size

        # Read labels
        cache = self.load_label(filenames)
        labels, shapes = zip(*cache.values())

        self.labels = list(labels)
        self.shapes = numpy.array(shapes, dtype=numpy.float64)

        self.filenames = list(cache.keys())  # update
        self.indices = range(len(shapes))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        index = self.indices[index]
        while True:
            if self.augment:
                # Load MOSAIC
                image, label = util.load_mosaic(self, index)
                # MixUp augmentation
                if random.random() < 0.1:
                    index = random.choice(self.indices)
                    mix_image1, mix_label1 = image, label
                    mix_image2, mix_label2 = util.load_mosaic(self, index)

                    image, label = util.mix_up(mix_image1, mix_label1, mix_image2, mix_label2)
                if not len(label):
                    index = random.choice(self.indices)
                    continue
                # HSV color-space
                util.augment_hsv(image)
                # Flip left-right
                if random.random() < 0.5:
                    image = numpy.fliplr(image)
                    if len(label):
                        label[:, 1:] = util.xy2wh(label[:, 1:], image.shape[1], image.shape[0])
                        label[:, 1] = 1 - label[:, 1]
                        label[:, 1:] = util.wh2xy(label[:, 1:], image.shape[1], image.shape[0])
            else:
                # Load image
                image, shape, (h, w) = util.load_image(self, index)

                # Letterbox
                image, ratio, pad = util.resize(image, self.input_size)

                label = self.labels[index].copy()
                if len(label):
                    label[:, 1:] = util.wh2xy(label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])
            # Convert HWC to CHW, BGR to RGB
            sample = image.transpose((2, 0, 1))[::-1]
            sample = numpy.ascontiguousarray(sample)
            if self.augment:
                target = {'boxes': torch.from_numpy(label[:, 1:].astype('float32')),
                          'labels': torch.from_numpy((label[:, :1].reshape(-1) + 1.).astype('int64'))}
                return torch.from_numpy(sample), target
            else:
                target = torch.zeros((len(label), 6))
                if len(label):
                    target[:, 1:] = torch.from_numpy(label)
                return torch.from_numpy(sample), target

    @staticmethod
    def collate_fn1(batch):
        samples, targets = zip(*batch)
        return torch.stack(samples, 0), targets

    @staticmethod
    def collate_fn2(batch):
        samples, targets = zip(*batch)
        for i, target in enumerate(targets):
            target[:, 0] = i
        return torch.stack(samples, 0), torch.cat(targets, 0)

    @staticmethod
    def load_label(filenames):
        x = {}
        for filename in filenames:
            try:
                # verify images
                image = Image.open(filename)
                image.verify()  # PIL verify
                shape = image.size  # image size
                assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} <10 pixels'
                assert image.format.lower() in FORMATS, f'invalid image format {image.format}'

                # verify labels
                a = f'{os.sep}images{os.sep}'
                b = f'{os.sep}labels{os.sep}'
                if os.path.isfile(b.join(filename.rsplit(a, 1)).rsplit('.', 1)[0] + '.txt'):
                    with open(b.join(filename.rsplit(a, 1)).rsplit('.', 1)[0] + '.txt') as f:
                        label = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        label = numpy.array(label, dtype=numpy.float32)
                    nl = len(label)
                    if nl:
                        assert label.shape[1] == 5, 'labels require 5 columns'
                        assert (label >= 0).all(), 'negative label values'
                        assert (label[:, 1:] <= 1).all(), 'non-normalized coordinates'
                        _, i = numpy.unique(label, axis=0, return_index=True)
                        if len(i) < nl:  # duplicate row check
                            label = label[i]  # remove duplicates
                    else:
                        label = numpy.zeros((0, 5), dtype=numpy.float32)
                else:
                    label = numpy.zeros((0, 5), dtype=numpy.float32)
                if filename:
                    x[filename] = [label, shape]
            except FileNotFoundError:
                pass
        return x
