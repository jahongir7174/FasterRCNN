import argparse
import copy
import glob
import math
import os

import numpy
import torch
import tqdm
import yaml

from nets import nn
from utils import util
from utils.dataset import input_fn


def learning_rate(params, epochs):
    def fn(x):
        return ((1 - math.cos(x * math.pi / epochs)) / 2) * (params['lrf'] - 1) + 1

    return fn


def train(params, args, device):
    epochs = 300
    util.init_seeds()

    model = nn.FastRCNN(len(params['names']) + 1)
    if args.local_rank == 0:
        util.model_info(copy.deepcopy(model).fuse(), args.image_size)
    model = model.to(device)
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, torch.nn.BatchNorm2d):
            pg0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):
            pg1.append(v.weight)

    optimizer = torch.optim.SGD(pg0, params['lr0'], params['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': params['weight_decay']})
    optimizer.add_param_group({'params': pg2})
    del pg0, pg1, pg2

    lr = learning_rate(params, epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr)

    filenames = glob.glob("../Dataset/COCO/labels/train2017/*.txt", recursive=True)
    loader = input_fn(filenames, args, params, True)
    # DDP mode
    if args.world_size > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    num_batch = len(loader)
    num_warmup = max(round(params['warmup_epochs'] * len(loader)), 1000)
    scheduler.last_epoch = -1
    amp_scale = torch.cuda.amp.GradScaler()

    best_fitness = 0.0
    for epoch in range(0, epochs):
        model.train()
        if args.world_size > 1:
            loader.sampler.set_epoch(epoch)
        p_bar = enumerate(loader)
        if args.local_rank == 0:
            print(('\n' + '%10s' * 3) % ('epoch', 'memory', 'loss'))
            p_bar = tqdm.tqdm(p_bar, total=num_batch)
        optimizer.zero_grad()
        for i, (images, target) in p_bar:
            n_iter = i + num_batch * epoch
            images = images.to(device, non_blocking=True).float() / 255.0
            target = [{k: v.to(device) for k, v in t.items()} for t in target]
            # Warmup
            if n_iter <= num_warmup:
                xi = [0, num_warmup]
                for j, x in enumerate(optimizer.param_groups):
                    if j == 2:
                        x['lr'] = numpy.interp(n_iter, xi,
                                               [params['warmup_bias_lr'],
                                                x['initial_lr'] * lr(epoch)])
                    else:
                        x['lr'] = numpy.interp(n_iter, xi,
                                               [0.0, x['initial_lr'] * lr(epoch)])
                    if 'momentum' in x:
                        fp = [params['warmup_momentum'], params['momentum']]
                        x['momentum'] = numpy.interp(n_iter, xi, fp)
            # Forward
            with torch.cuda.amp.autocast():
                loss = model(images, target)
                loss = sum(v for v in loss.values())
            # Backward
            amp_scale.scale(loss).backward()
            # Optimize
            amp_scale.step(optimizer)
            amp_scale.update()
            optimizer.zero_grad()
            if args.local_rank == 0:
                memory = '%.3gG' % (torch.cuda.memory_reserved() / 1E9)
                s = ('%10s' * 2 + '%10.4g') % ('%g/%g' % (epoch + 1, epochs), memory, loss.detach())
                p_bar.set_description(s)
        # Scheduler
        scheduler.step()
        # DDP process 0 or single-GPU
        if args.local_rank == 0:
            mp, mr, map50, mean_ap = test(model.module, args, params)
            current = util.fitness(numpy.array([mp, mr, map50, mean_ap]).reshape(1, -1))
            if current > best_fitness:
                best_fitness = current
            save = {'model': copy.deepcopy(model.module).half()}
            torch.save(save, 'weights/last.pt')
            if best_fitness == current:
                torch.save(save, 'weights/best.pt')
            del save
    if args.local_rank == 0:
        util.disable_grad('weights/last.pt')
        util.disable_grad('weights/best.pt')
    if args.world_size > 1:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


def test(model=None, args=None, params=None):
    if model is not None:
        device = next(model.parameters()).device
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = torch.load('weights/best.pt', device)['model'].float().eval()

    model.eval()

    iou_v = torch.linspace(0.5, 0.95, 10).to(device)
    n_iou = iou_v.numel()

    filenames = glob.glob("../Dataset/COCO/labels/val2017/*.txt", recursive=True)
    loader = input_fn(filenames, args, params)
    seen = 0
    s = ('%10s' * 3) % ('pre', 'rec', 'mAP')
    p, r, f1, mp, mr, map50, mean_ap, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    stats, ap, ap_class = [], [], []
    for images, target in tqdm.tqdm(loader, desc=s):
        images = images.to(device, non_blocking=True).float() / 255.0
        target = target.to(device)
        _, _, height, width = images.shape
        with torch.no_grad():
            t = util.time_synchronized()
            inf_output = model(images)
            t0 += util.time_synchronized() - t
            t = util.time_synchronized()
            t1 += util.time_synchronized() - t
        output = []
        for inf in inf_output:
            output.append(torch.cat([inf['boxes'],
                                     inf['scores'].unsqueeze(-1),
                                     (inf['labels'] - 1).unsqueeze(-1)], -1))
        for si, pred in enumerate(output):
            seen += 1
            label = target[target[:, 0] == si, 1:]
            t_cls = label[:, 0].tolist() if len(label) else []

            if len(pred) == 0:
                if len(label):
                    stat = (torch.zeros(0, n_iou, dtype=torch.bool),
                            torch.Tensor(), torch.Tensor(), t_cls)
                    stats.append(stat)
                continue

            pred_n = pred.clone()
            correct = torch.zeros(pred.shape[0], n_iou, dtype=torch.bool, device=device)
            if len(label):
                detected = []
                t_cls_tensor = label[:, 0]
                t_box = label[:, 1:5]

                for cls in torch.unique(t_cls_tensor):
                    ti = (cls == t_cls_tensor).nonzero(as_tuple=False).view(-1)
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)

                    if pi.shape[0]:
                        iou_list, i = util.box_iou(pred_n[pi, :4], t_box[ti]).max(1)

                        detected_set = set()
                        for j in (iou_list > iou_v[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = iou_list[j] > iou_v
                                if len(detected) == len(label):
                                    break

            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), t_cls))

    stats = [numpy.concatenate(x, 0) for x in zip(*stats)]
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = util.ap_per_class(*stats)
        p, r, ap50, ap = p[:, 0], r[:, 0], ap[:, 0], ap.mean(1)
        mp, mr, map50, mean_ap = p.mean(), r.mean(), ap50.mean(), ap.mean()

    print('%10.3g' * 3 % (mp, mr, mean_ap))

    if model is None:
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1))
        s = f'Speed: {t[0]:.1f}/{t[1]:.1f}/{t[2]:.1f} ms inference/nms/total'
        print(f'{s} per image at batch-size {args.batch_size}')

    model.float()
    return mp, mr, map50, mean_ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=48)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda:0')
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    if args.world_size > 1:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    with open(os.path.join('utils', 'args.yaml')) as f:
        params = yaml.safe_load(f)
    if args.local_rank == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')
    train(params, args, device)


if __name__ == '__main__':
    main()
