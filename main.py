import argparse
import copy
import os
import warnings

import numpy
import torch
import tqdm
import yaml
from timm import utils
from torch.utils import data

from nets import nn
from utils import util
from utils.dataset import Dataset

warnings.filterwarnings("ignore")


def learning_rate(args, params):
    def fn(x):
        return (1 - x / args.epochs) * (1.0 - params['lrf']) + params['lrf']

    return fn


def train(args, params):
    # Model
    model = nn.faster_r_cnn_n(len(params['names'].values()) + 1).cuda()

    # Optimizer
    accumulate = max(round(64 / (args.batch_size * args.world_size)), 1)
    params['weight_decay'] *= args.batch_size * args.world_size * accumulate / 64

    p = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
            p[2].append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d):
            p[1].append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
            p[0].append(v.weight)

    optimizer = torch.optim.SGD(p[2], params['lr0'], params['momentum'], nesterov=True)

    optimizer.add_param_group({'params': p[0], 'weight_decay': params['weight_decay']})
    optimizer.add_param_group({'params': p[1]})
    del p

    # Scheduler
    lr = learning_rate(args, params)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr, last_epoch=-1)

    # EMA
    ema = util.EMA(model) if args.local_rank == 0 else None

    filenames = []
    with open('../Dataset/COCO/train2017.txt') as reader:
        for filename in reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append('../Dataset/COCO/images/train2017/' + filename)

    if args.world_size > 1:
        dataset = Dataset(filenames, args.input_size, True)
        sampler = data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
        dataset = Dataset(filenames, args.input_size, True)

    loader = data.DataLoader(dataset, args.batch_size, sampler is None, sampler,
                             num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn1)

    if args.world_size > 1:
        # DDP mode
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)

    # Start training
    best = 0
    num_batch = len(loader)
    amp_scale = torch.cuda.amp.GradScaler()
    num_warmup = max(round(params['warmup_epochs'] * num_batch), 1000)
    for epoch in range(args.epochs):
        model.train()

        m_loss = util.AverageMeter()
        if args.world_size > 1:
            sampler.set_epoch(epoch)
        p_bar = enumerate(loader)
        if args.local_rank == 0:
            print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
        if args.local_rank == 0:
            p_bar = tqdm.tqdm(p_bar, total=num_batch)  # progress bar

        optimizer.zero_grad()

        for i, (samples, targets) in p_bar:
            x = i + num_batch * epoch  # number of iterations
            samples = samples.cuda().float() / 255
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            # Warmup
            if x <= num_warmup:
                xp = [0, num_warmup]
                fp = [1, 64 / (args.batch_size * args.world_size)]
                accumulate = max(1, numpy.interp(x, xp, fp).round())
                for j, y in enumerate(optimizer.param_groups):
                    if j == 0:
                        fp = [params['warmup_bias_lr'], y['initial_lr'] * lr(epoch)]
                    else:
                        fp = [0.0, y['initial_lr'] * lr(epoch)]
                    y['lr'] = numpy.interp(x, xp, fp)
                    if 'momentum' in y:
                        fp = [params['warmup_momentum'], params['momentum']]
                        y['momentum'] = numpy.interp(x, xp, fp)

            # Forward
            with torch.cuda.amp.autocast():
                loss = model(samples, targets)  # forward
            loss = sum(v for v in loss.values())
            # Backward
            amp_scale.scale(loss).backward()

            # Optimize
            if x % accumulate == 0:
                amp_scale.step(optimizer)  # optimizer.step
                amp_scale.update()
                optimizer.zero_grad()
                if ema:
                    ema.update(model)

            if args.world_size > 1:
                loss = utils.reduce_tensor(loss.data, args.world_size)

            m_loss.update(loss.item(), samples.size(0))
            # Log
            if args.local_rank == 0:
                memory = f'{torch.cuda.memory_reserved() / 1E9:.3g}G'  # (GB)
                s = ('%10s' * 2 + '%10.4g') % (f'{epoch + 1}/{args.epochs}', memory, m_loss.avg)
                p_bar.set_description(s)
        # Scheduler
        scheduler.step()

        if args.local_rank == 0:
            # mAP
            last = test(args, ema.ema)

            # Update best mAP
            if last > best:
                best = last

            # Save model
            ckpt = {'model': copy.deepcopy(ema.ema).half()}

            # Save last, best and delete
            torch.save(ckpt, './weights/last.pt')
            if best == last:
                torch.save(ckpt, './weights/best.pt')
            del ckpt

    if args.local_rank == 0:
        util.strip_optimizer('./weights/best.pt')  # strip optimizers
        util.strip_optimizer('./weights/last.pt')  # strip optimizers

    torch.cuda.empty_cache()


@torch.no_grad()
def test(args, model=None):
    filenames = []
    with open('../Dataset/COCO/val2017.txt') as reader:
        for filename in reader.readlines():
            filename = filename.rstrip().split('/')[-1]
            filenames.append('../Dataset/COCO/images/val2017/' + filename)
    dataset = Dataset(filenames, args.input_size, False)
    loader = data.DataLoader(dataset, 4, False, num_workers=4,
                             pin_memory=True, collate_fn=Dataset.collate_fn2)

    if model is None:
        model = torch.load('./weights/best.pt', map_location='cuda')['model'].float()

    model.eval()
    model.half()

    # Configure
    iou_v = torch.linspace(0.5, 0.95, 10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    s = ('%10s' * 3) % ('precision', 'recall', 'mAP')
    p, r, f1, mp, mr, map50, mean_ap = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    stats, ap, ap_class = [], [], []
    p_bar = tqdm.tqdm(loader, desc=s)
    for samples, targets in p_bar:
        samples = samples.cuda()
        targets = targets.cuda()
        samples = samples.half()  # uint8 to fp16/32
        samples = samples / 255  # 0 - 255 to 0.0 - 1.0
        _, _, height, width = samples.shape  # batch size, channels, height, width

        # Inference
        outputs = []
        inf_outputs = model(samples)
        for inf_output in inf_outputs:
            outputs.append(torch.cat([inf_output['boxes'],
                                      inf_output['scores'].unsqueeze(-1),
                                      (inf_output['labels'] - 1).unsqueeze(-1)], -1))

        # Metrics
        for si, output in enumerate(outputs):
            labels = targets[targets[:, 0] == si, 1:]
            num_target = labels.shape[0]  # number of labels
            num_output = output.shape[0]  # number of predictions
            correct = torch.zeros(num_output, n_iou, dtype=torch.bool).cuda()  # init

            if num_output == 0:
                if num_target:
                    stats.append((correct, *torch.zeros((3, 0)).cuda()))
                continue

            detections = output.clone()

            # Evaluate
            if num_target:
                tbox = labels[:, 1:5].clone()  # target boxes

                correct = numpy.zeros((detections.shape[0], iou_v.shape[0]))
                correct = correct.astype(bool)

                t_tensor = torch.cat((labels[:, 0:1], tbox), 1)
                iou = util.box_iou(t_tensor[:, 1:], detections[:, :4])
                correct_class = t_tensor[:, 0:1] == detections[:, 5]
                for i in range(len(iou_v)):
                    x = torch.where((iou >= iou_v[i]) & correct_class)
                    if x[0].shape[0]:
                        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
                        if x[0].shape[0] > 1:
                            matches = matches[matches[:, 2].argsort()[::-1]]
                            matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), i] = True
                correct = torch.tensor(correct, dtype=torch.bool, device=iou_v.device)
            stats.append((correct, output[:, 4], output[:, 5], labels[:, 0]))

    # Compute metrics
    stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        tp, fp, p, r, f1, ap, ap_class = util.ap_per_class(*stats)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, mean_ap = p.mean(), r.mean(), ap50.mean(), ap.mean()

    # Print results
    print('%10.3g' * 3 % (mp, mr, mean_ap))

    # Return results
    model.float()  # for training
    return mean_ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-size', default=640, type=int)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')

    args = parser.parse_args()

    args.local_rank = int(os.getenv('LOCAL_RANK', 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))

    if args.world_size > 1:
        torch.cuda.set_device(device=args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    util.setup_seed()
    util.setup_multi_processes()

    with open(os.path.join('utils', 'args.yaml'), errors='ignore') as f:
        params = yaml.safe_load(f)

    if args.train:
        train(args, params)
    if args.test:
        test(args)


if __name__ == '__main__':
    main()
