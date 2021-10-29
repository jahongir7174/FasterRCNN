import contextlib
import copy
import random
import time

import numpy
import torch


@contextlib.contextmanager
def distributed_manager(local_rank: int):
    if local_rank != 0:
        torch.distributed.barrier()
    yield
    if local_rank == 0:
        torch.distributed.barrier()


def init_seeds():
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def model_params(model, weight_decay=1e-5):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def model_info(model, img_size):
    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)

    try:
        from thop import profile
        flops = profile(copy.deepcopy(model),
                        inputs=(torch.zeros(1, 3, img_size, img_size),),
                        verbose=False)[0] / 1E9 * 2
        fs = ', %.1f GFLOPS' % flops
    except (ImportError, Exception):
        fs = ''
    s = f"{len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}"
    print(f"Model Summary: {s}")


def fitness(x):
    w = [0.0, 0.0, 0.0, 1.0]
    return (x[:, :4] * w).sum(1)


def compute_ap(recall, precision):
    # Append sentinel values to beginning and end
    m_rec = recall  # np.concatenate(([0.], recall, [recall[-1] + 1E-3]))
    m_pre = precision  # np.concatenate(([0.], precision, [0.]))

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


def ap_per_class(tp, conf, pred_cls, target_cls):
    # Sort by object-ness
    i = numpy.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = numpy.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    px, py = numpy.linspace(0, 1, 1000), []  # for plotting
    pr_score = 0.1  # score to evaluate P and R https://github.com/ultralytics/yolov3/issues/898
    s = [unique_classes.shape[0], tp.shape[1]]  # number class, number iou thresholds (i.e. 10 for mAP0.5...0.95)
    ap, p, r = numpy.zeros(s), numpy.zeros(s), numpy.zeros(s)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            # r at pr_score, negative x, xp because xp decreases
            r[ci] = numpy.interp(-pr_score, -conf[i], recall[:, 0])

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve
            p[ci] = numpy.interp(-pr_score, -conf[i], precision[:, 0])  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], m_pre, m_rec = compute_ap(recall[:, j], precision[:, j])
                if j == 0:
                    py.append(numpy.interp(px, m_rec, m_pre))  # precision at mAP@0.5

    # Compute F1 score (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype('int32')


def xyn2xy(x, w=640, h=640, pad_w=0, pad_h=0):
    y = numpy.copy(x)
    y[:, 0] = w * x[:, 0] + pad_w  # top left x
    y[:, 1] = h * x[:, 1] + pad_h  # top left y
    return y


def whn2xy(x, w=640, h=640, pad_w=0, pad_h=0):
    y = numpy.copy(x)
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + pad_w  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + pad_h  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + pad_w  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + pad_h  # bottom right y
    return y


def segments2boxes(segments):
    boxes = []
    for s in segments:
        x, y = s.T
        boxes.append([x.min(), y.min(), x.max(), y.max()])
    x = numpy.array(boxes)
    y = numpy.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def segment2box(segment, width=640, height=640):
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    if any(x):
        return numpy.array([x.min(), y.min(), x.max(), y.max()])
    else:
        return numpy.zeros((1, 4))


def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        x = numpy.linspace(0, len(s) - 1, n)
        xp = numpy.arange(len(s))
        segment = [numpy.interp(x, xp, s[:, i]) for i in range(2)]
        segments[i] = numpy.concatenate(segment).reshape(2, -1).T
    return segments


def box_ioa(box1, box2, eps=1E-7):
    box2 = box2.transpose()

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # Intersection area
    inter_area = (numpy.minimum(b1_x2, b2_x2) - numpy.maximum(b1_x1, b2_x1)).clip(0) * \
                 (numpy.minimum(b1_y2, b2_y2) - numpy.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def box_iou(box1, box2):
    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])
    lt = torch.max(box1[:, None, :2], box2[:, :2])
    inter = (rb - lt).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)


def disable_grad(filename):
    x = torch.load(filename, torch.device('cpu'))
    for param in x['model'].parameters():
        param.requires_grad = False
    torch.save(x, filename)
