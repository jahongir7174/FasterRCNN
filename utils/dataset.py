import math
import os
import random
from itertools import repeat
from multiprocessing.pool import ThreadPool

import cv2
import numpy
import torch
from torch.utils import data

from utils import util


def input_fn(filenames, args, params=None, augment=False):
    if args.world_size > 1 and augment:
        with util.distributed_manager(args.local_rank):
            dataset = Dataset(filenames, args.image_size, augment, params)
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        sampler = None
        dataset = Dataset(filenames, args.image_size, augment, params)
        args.batch_size = 16
    collate_fn = Dataset.collate_fn1 if augment else Dataset.collate_fn2
    loader = data.DataLoader(dataset, args.batch_size, sampler=sampler,
                             num_workers=8, pin_memory=True, collate_fn=collate_fn)
    return loader


class Dataset(data.Dataset):
    def __init__(self, filenames, image_size=640, augment=False, params=None):
        self.params = params
        self.augment = augment
        self.image_size = image_size
        self.mosaic_border = [-image_size // 2, -image_size // 2]

        cache = self.cache_labels(filenames)
        labels, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.filenames = list(cache.keys())

        filenames_npy = [x.replace('jpg', 'npy') for x in self.filenames]
        if not all([os.path.exists(x) for x in filenames_npy]):
            results = ThreadPool(8).imap(lambda x: load_image(*x),
                                         zip(repeat(self), range(len(self.filenames))))
            for i, result in enumerate(results):
                numpy.save(filenames_npy[i], result[0])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        if self.augment:
            # Load mosaic
            image, label = load_mosaic(self, index)
            # MixUp augmentation
            if random.random() < self.params['mixup']:
                mix_up_index = random.randint(0, len(self.filenames) - 1)
                image, label = mix_up(image, label, *load_mosaic(self, mix_up_index))
            # HSV color-space
            augment_hsv(image)
        else:
            # Load image
            image, (h, w) = load_image(self, index)
            image, ratio, pad = resize(image, self.image_size)
            label = self.labels[index].copy()
            if label.size:
                label[:, 1:] = util.whn2xy(label[:, 1:], ratio[0] * w, ratio[1] * h, pad[0], pad[1])
        # HxWxC -> CxHxW, BGR -> RGB
        image = image.transpose((2, 0, 1))[::-1]
        image = numpy.ascontiguousarray(image)
        label1 = {'boxes': torch.from_numpy(label[:, 1:].astype('float32')),
                  'labels': torch.from_numpy((label[:, :1].reshape(-1) + 1.).astype('int64'))}
        label2 = torch.zeros((len(label), 6))
        label2[:, 1:] = torch.from_numpy(label)
        return torch.from_numpy(image), label1 if self.augment else label2

    @staticmethod
    def cache_labels(filenames):
        x = {}
        for filename in filenames:
            try:
                with open(filename, 'r') as f:
                    y = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    classes = numpy.array([x[0] for x in y], dtype=numpy.float32)
                    segment = [numpy.array(x[1:], numpy.float32).reshape(-1, 2) for x in y]
                    y = numpy.concatenate((classes.reshape(-1, 1),
                                           util.segments2boxes(segment)), 1)
                    y = numpy.array(y, dtype=numpy.float32)
                if not len(y):
                    y = numpy.zeros((0, 5), dtype=numpy.float32)
                filename = filename.replace('labels', 'images').replace('txt', 'jpg')
                x[filename] = [y, segment]
            except FileNotFoundError:
                pass
        return x

    @staticmethod
    def collate_fn1(batch):
        image, label = zip(*batch)
        return torch.stack(image, 0), label

    @staticmethod
    def collate_fn2(batch):
        image, label = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i
        return torch.stack(image, 0), torch.cat(label, 0)


def load_image(self, index):
    filename = self.filenames[index]
    if os.path.exists(filename.replace('jpg', 'npy')):
        image = numpy.load(filename.replace('jpg', 'npy'))
    else:
        image = cv2.imread(filename)  # BGR
    h, w = image.shape[:2]
    r = self.image_size / max(h, w)  # ratio
    if r != 1:  # if sizes are not equal
        image = cv2.resize(image, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
    return image, image.shape[:2]


def load_mosaic(self, index):
    label4, segment4 = [], []
    s = self.image_size
    yc, xc = [int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]
    indices = [index] + random.choices(range(len(self.filenames)), k=3)
    random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        image, (h, w) = load_image(self, index)
        if i == 0:  # top left
            image4 = numpy.full((s * 2, s * 2, image.shape[2]), 114, dtype=numpy.uint8)
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
        image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
        pad_w = x1a - x1b
        pad_h = y1a - y1b
        label, segment = self.labels[index].copy(), self.segments[index].copy()
        if label.size:
            label[:, 1:] = util.whn2xy(label[:, 1:], w, h, pad_w, pad_h)
            segment = [util.xyn2xy(x, w, h, pad_w, pad_h) for x in segment]
        label4.append(label)
        segment4.extend(segment)
    # Concat/clip labels
    label4 = numpy.concatenate(label4, 0)
    for x in (label4[:, 1:], *segment4):
        numpy.clip(x, 0, 2 * s, out=x)
    # Augment
    image4, label4, segment4 = copy_paste(image4, label4, segment4)
    image4, label4 = random_perspective(image4, label4, segment4,
                                        border=self.mosaic_border)  # border to remove
    return image4, label4


def augment_hsv(image, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4):
    # HSV color-space augmentation
    r = numpy.random.uniform(-1, 1, 3) * [hsv_h, hsv_s, hsv_v] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    x = numpy.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype('uint8')
    lut_sat = numpy.clip(x * r[1], 0, 255).astype('uint8')
    lut_val = numpy.clip(x * r[2], 0, 255).astype('uint8')

    im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=image)


def resize(image, new_shape, color=(114, 114, 114)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]  # current shape [height, width]
    new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1], 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_pad[0], new_shape[0] - new_pad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_pad:  # resize
        image = cv2.resize(image, new_pad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, ratio, (dw, dh)


def random_perspective(image, label=(), segment=(),
                       degrees=0., translate=.1, scale=.5,
                       shear=0., perspective=0.0, border=(0, 0)):
    h = image.shape[0] + border[0] * 2  # shape(h,w,c)
    w = image.shape[1] + border[1] * 2
    # Center
    c_gain = numpy.eye(3)
    c_gain[0, 2] = -image.shape[1] / 2  # x translation (pixels)
    c_gain[1, 2] = -image.shape[0] / 2  # y translation (pixels)
    # Perspective
    p_gain = numpy.eye(3)
    p_gain[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    p_gain[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)
    # Rotation and Scale
    r_gain = numpy.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    r_gain[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    s_gain = numpy.eye(3)
    s_gain[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    s_gain[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    t_gain = numpy.eye(3)
    t_gain[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * w  # x translation (pixels)
    t_gain[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * h  # y translation (pixels)

    # Combined rotation matrix, order of operations (right to left) is IMPORTANT
    matrix = t_gain @ s_gain @ r_gain @ p_gain @ c_gain
    if (border[0] != 0) or (border[1] != 0) or (matrix != numpy.eye(3)).any():  # image changed
        if perspective:
            image = cv2.warpPerspective(image, matrix, dsize=(w, h), borderValue=(114, 114, 114))
        else:  # affine
            image = cv2.warpAffine(image, matrix[:2], dsize=(w, h), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(label)
    if n:
        new = numpy.zeros((n, 4))
        segment = util.resample_segments(segment)
        for i, seg in enumerate(segment):
            xy = numpy.ones((len(seg), 3))
            xy[:, :2] = seg
            xy = xy @ matrix.T  # transform
            # perspective rescale or affine
            xy = xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]
            # clip
            new[i] = util.segment2box(xy, w, h)
        # filter candidates
        i = box_candidates(label[:, 1:5].T * s, new.T, area_thr=0.01)
        label = label[i]
        label[:, 1:5] = new[i]
    return image, label


def copy_paste(image, label, segment, p=0.):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177
    n = len(segment)
    if p and n:
        h, w, c = image.shape  # height, width, channels
        image_cp = numpy.zeros(image.shape, numpy.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = label[j], segment[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = util.box_ioa(box, label[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                label = numpy.concatenate((label, [[l[0], *box]]), 0)
                segment.append(numpy.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(image_cp,
                                 [segment[j].astype(numpy.int32)],
                                 -1, (255, 255, 255), cv2.FILLED)

        result = cv2.bitwise_and(src1=image, src2=image_cp)
        result = cv2.flip(result, 1)  # augment segments (flip left-right)
        i = result > 0  # pixels to replace
        image[i] = result[i]

    return image, label, segment


def mix_up(image, label, image2, label2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = numpy.random.beta(32.0, 32.0)  # mix-up ratio, alpha=beta=32.0
    image = (image * r + image2 * (1 - r)).astype(numpy.uint8)
    label = numpy.concatenate((label, label2), 0)
    return image, label


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = numpy.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)
