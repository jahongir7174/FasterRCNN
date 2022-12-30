import copy
import math
import random
import time
from os import environ
from platform import system

import cv2
import numpy
import torch
import torchvision


def setup_seed():
    """
    Setup random seed.
    """
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def load_image(self, index):
    image = cv2.imread(self.filenames[index])
    h, w = image.shape[:2]
    r = self.input_size / max(h, w)
    if r != 1:
        resample = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
        image = cv2.resize(image, dsize=(int(w * r), int(h * r)), interpolation=resample)
    return image, (h, w), image.shape[:2]


def load_mosaic(self, index):
    label4 = []
    image4 = numpy.full((self.input_size * 2, self.input_size * 2, 3), 0, dtype=numpy.uint8)
    y1a, y2a, x1a, x2a, y1b, y2b, x1b, x2b = (None, None, None, None, None, None, None, None)

    xc = int(random.uniform(self.input_size // 2, 2 * self.input_size - self.input_size // 2))
    yc = int(random.uniform(self.input_size // 2, 2 * self.input_size - self.input_size // 2))

    indices = [index] + random.choices(self.indices, k=3)
    random.shuffle(indices)

    for i, index in enumerate(indices):
        # Load image
        image, _, (h, w) = load_image(self, index)
        if i == 0:  # top left
            x1a = max(xc - w, 0)
            y1a = max(yc - h, 0)
            x2a = xc
            y2a = yc
            x1b = w - (x2a - x1a)
            y1b = h - (y2a - y1a)
            x2b = w
            y2b = h
        elif i == 1:  # top right
            x1a = xc
            y1a = max(yc - h, 0)
            x2a = min(xc + w, self.input_size * 2)
            y2a = yc
            x1b = 0
            y1b = h - (y2a - y1a)
            x2b = min(w, x2a - x1a)
            y2b = h
        elif i == 2:  # bottom left
            x1a = max(xc - w, 0)
            y1a = yc
            x2a = xc
            y2a = min(self.input_size * 2, yc + h)
            x1b = w - (x2a - x1a)
            y1b = 0
            x2b = w
            y2b = min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a = xc
            y1a = yc
            x2a = min(xc + w, self.input_size * 2)
            y2a = min(self.input_size * 2, yc + h)
            x1b = 0
            y1b = 0
            x2b = min(w, x2a - x1a)
            y2b = min(y2a - y1a, h)

        image4[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
        pad_w = x1a - x1b
        pad_h = y1a - y1b

        # Labels
        label = self.labels[index].copy()
        if len(label):
            label[:, 1:] = wh2xy(label[:, 1:], w, h, pad_w, pad_h)
        label4.append(label)

    # Concat/clip labels
    label4 = numpy.concatenate(label4, 0)
    for x in label4[:, 1:]:
        numpy.clip(x, 0, 2 * self.input_size, out=x)

    # Augment
    image4, label4 = random_perspective(image4, label4, self.input_size)

    return image4, label4


def augment_hsv(image):
    # HSV color-space augmentation
    r = numpy.random.uniform(-1, 1, 3) * [.015, .7, .4] + 1
    h, s, v = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))

    x = numpy.arange(0, 256, dtype=r.dtype)
    lut_h = ((x * r[0]) % 180).astype('uint8')
    lut_s = numpy.clip(x * r[1], 0, 255).astype('uint8')
    lut_v = numpy.clip(x * r[2], 0, 255).astype('uint8')

    im_hsv = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=image)  # no return needed


def resize(image, input_size):
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    r = min(1.0, input_size / shape[0], input_size / shape[1])

    # Compute padding
    pad = int(round(shape[1] * r)), int(round(shape[0] * r))
    w = (input_size - pad[0]) / 2
    h = (input_size - pad[1]) / 2

    if shape[::-1] != pad:  # resize
        image = cv2.resize(image, pad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
    left, right = int(round(w - 0.1)), int(round(w + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)  # add border
    return image, (r, r), (w, h)


def box_candidates(box1, box2):
    # box1(4,n), box2(4,n)
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    aspect_ratio = numpy.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (w2 > 2) & (h2 > 2) & (w2 * h2 / (w1 * h1 + 1e-16) > 0.1) & (aspect_ratio < 100)  # candidates


def random_perspective(samples, targets, input_size):
    # Center
    center = numpy.eye(3)
    center[0, 2] = -float(input_size)  # x translation (pixels)
    center[1, 2] = -float(input_size)  # y translation (pixels)

    # Perspective
    perspective = numpy.eye(3)
    perspective[2, 0] = random.uniform(-0.0, 0.0)  # x perspective (about y)
    perspective[2, 1] = random.uniform(-0.0, 0.0)  # y perspective (about x)

    # Rotation and Scale
    rotation = numpy.eye(3)
    a = random.uniform(-0, 0)
    s = random.uniform(1 - 0.5, 1 + 0.5)
    rotation[:2] = cv2.getRotationMatrix2D(center=(0, 0), angle=a, scale=s)

    # Shear
    shear = numpy.eye(3)
    shear[0, 1] = math.tan(random.uniform(-0.0, 0.0) * math.pi / 180)  # x shear (deg)
    shear[1, 0] = math.tan(random.uniform(-0.0, 0.0) * math.pi / 180)  # y shear (deg)

    # Translation
    translation = numpy.eye(3)
    translation[0, 2] = random.uniform(0.5 - 0.1, 0.5 + 0.1) * input_size  # x translation (pixels)
    translation[1, 2] = random.uniform(0.5 - 0.1, 0.5 + 0.1) * input_size  # y translation (pixels)

    # Combined rotation matrix, order of operations (right to left) is IMPORTANT
    matrix = translation @ shear @ rotation @ perspective @ center
    # image changed
    samples = cv2.warpAffine(samples, matrix[:2], dsize=(input_size, input_size))

    n = len(targets)
    if n:
        xy = numpy.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ matrix.T  # transform
        xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = numpy.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, input_size)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, input_size)

        # filter candidates
        i = box_candidates(targets[:, 1:5].T * s, new.T)
        targets = targets[i]
        targets[:, 1:5] = new[i]

    return samples, targets


def mix_up(image1, label1, image2, label2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = numpy.random.beta(32.0, 32.0)  # mix-up ratio, alpha=beta=32.0
    image = (image1 * r + image2 * (1 - r)).astype(numpy.uint8)
    label = numpy.concatenate((label1, label2), 0)
    return image, label


def check_anchors(dataset, model, args, params):
    shapes = dataset.shapes / dataset.shapes.max(1, keepdims=True)
    shapes = shapes * args.input_size

    wh = []
    for shape, label in zip(shapes, dataset.labels):
        wh.append(label[:, 3:5] * shape)
    wh = torch.tensor(numpy.concatenate(wh)).float()

    def metric(k):
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        a = (x.max(1)[0] > 1 / params['anchor_t']).float().mean()
        b = (x > 1 / params['anchor_t']).float().sum(1).mean()
        return a, b

    m = model.head
    stride = m.stride.to(m.anchors.device).view(-1, 1, 1)  # model strides
    anchors = m.anchors.clone() * stride  # current anchors
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    if args.local_rank == 0:
        print(f'\n{aat:.2f} Anchors/Target, {bpr:.3f} Best Possible Recall (BPR). ')


def clip(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale(coords, shape1, shape2, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(shape1[0] / shape2[0], shape1[1] / shape2[1])  # gain  = old / new
        pad = (shape1[1] - shape2[1] * gain) / 2, (shape1[0] - shape2[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip(coords, shape2)
    return coords


def xy2wh(x, w, h):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    clip(x, (h - 1E-3, w - 1E-3))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y


def wh2xy(x, w, h, pad_w=0, pad_h=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # bottom right y
    return y


def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter)


def non_max_suppression(prediction, confidence=0.25, iou_threshold=0.45):
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > confidence  # candidates

    # Checks
    assert 0 <= confidence <= 1, f'Invalid Confidence threshold {confidence}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_threshold <= 1, f'Invalid IoU {iou_threshold}, valid values are between 0.0 and 1.0'

    # Settings
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.3 + 0.03 * bs  # seconds to quit after
    max_det = 300
    redundant = True  # require redundant detections
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = x[:, :4].clone()
        box[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        box[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        box[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        box[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

        # Detections matrix nx6 (xyxy, conf, cls)
        if nc > 1:
            i, j = (x[:, 5:] > confidence).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > confidence]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if nc == 1 else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_threshold  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


def smooth(y, f=0.05):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = numpy.ones(nf // 2)  # ones padding
    yp = numpy.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return numpy.convolve(yp, numpy.ones(nf) / nf, mode='valid')  # y-smoothed


def compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    m_rec = numpy.concatenate(([0.0], recall, [1.0]))
    m_pre = numpy.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    m_pre = numpy.flip(numpy.maximum.accumulate(numpy.flip(m_pre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = numpy.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = numpy.trapz(numpy.interp(x, m_rec, m_pre), x)  # integrate
    else:  # 'continuous'
        i = numpy.where(m_rec[1:] != m_rec[:-1])[0]  # points where x axis (recall) changes
        ap = numpy.sum((m_rec[i + 1] - m_rec[i]) * m_pre[i + 1])  # area under curve

    return ap, m_pre, m_rec


def ap_per_class(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Object-ness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by object-ness
    i = numpy.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes, nt = numpy.unique(target_cls, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = numpy.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = numpy.zeros((nc, tp.shape[1])), numpy.zeros((nc, 1000)), numpy.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = nt[ci]  # number of labels
        n_p = i.sum()  # number of predictions
        if n_p == 0 or n_l == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)  # recall curve
        r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)

    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    return tp, fp, p, r, f1, ap, unique_classes.astype(int)


def strip_optimizer(f='best.pt'):
    x = torch.load(f, map_location=torch.device('cpu'))
    x['model'].half()  # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, f)


class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()


class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num
