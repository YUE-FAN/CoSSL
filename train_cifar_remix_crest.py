"""
This script is based on train_cifar_fix.py

Here I implement CReST for ReMixMatch.

Differences:
- train_labeled_set uses strong DA
- SemiLoss is different
- remove num_samples_per_class += N_SAMPLES_PER_CLASS_T
- logger names are changed
- add interleave
- add scheduler in train() from train_mix.py
- replace fixmatch hypers with remixmatch hypers

Examples:
python train_remix_crest.py --dataset cifar10 --ratio 2 --num_max 1500 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 64 --val-iteration 1024 --out ./results/cifar10/remixmatch/crest/wrn28_N1500_r50_seed1/ --manualSeed 1 --gpu 0

python train_remix_crest.py --dataset cifar10 --ratio 2 --num_max 1500 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 64 --val-iteration 1024 --out ./results/cifar10/remixmatch/crest/wrn28_N1500_r100_seed1 --manualSeed 1 --gpu 0

python train_remix_crest.py --dataset cifar10 --ratio 2 --num_max 1500 --imb_ratio_l 150 --imb_ratio_u 150 --epoch 64 --val-iteration 1024 --out ./results/cifar10/remixmatch/crest/wrn28_N1500_r150_seed1 --manualSeed 1 --gpu 0

python train_remix_crest.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 20 --imb_ratio_u 20 --epoch 128 --val-iteration 1024 --out ./results/cifar100/remixmatch/crest/wrn28_N150_r20_seed1 --manualSeed 1 --gpu 0

python train_remix_crest.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 50 --imb_ratio_u 50 --epoch 128 --val-iteration 1024 --out ./results/cifar100/remixmatch/crest/wrn28_N150_r50_seed1 --manualSeed 1 --gpu 0

python train_remix_crest.py --dataset cifar100 --ratio 2 --num_max 150 --imb_ratio_l 100 --imb_ratio_u 100 --epoch 128 --val-iteration 1024 --out ./results/cifar100/remixmatch/crest/wrn28_N150_r100_seed1 --manualSeed 1 --gpu 0

"""
from __future__ import print_function

import argparse
import os
import json
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import time
import random
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
import torchvision
from PIL import Image

import models.wrn as models
from dataset.remix_cifar10 import transform_train as tfms_w_cifar10
from dataset.remix_cifar10 import transform_strong as tfms_s_cifar10
from dataset.remix_cifar100 import transform_train as tfms_w_cifar100
from dataset.remix_cifar100 import transform_strong as tfms_s_cifar100

from dataset.remix_cifar10 import transform_val as tfms_test_cifar10
from dataset.remix_cifar100 import transform_val as tfms_test_cifar100

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig

parser = argparse.ArgumentParser(description='PyTorch ReMixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=64, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,  # 0.002
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

# Method options
parser.add_argument('--dataset', type=str, default='cifar10',
                        help='cifar10 or cifar100')
parser.add_argument('--num_max', type=int, default=1500,
                        help='Number of samples in the maximal class')
parser.add_argument('--ratio', type=float, default=2.0,
                        help='Relative size between labeled and unlabeled data')
parser.add_argument('--imb_ratio_l', type=int, default=100,
                        help='Imbalance ratio for labeled data')
parser.add_argument('--imb_ratio_u', type=int, default=100,
                        help='Imbalance ratio for unlabeled data')
parser.add_argument('--step', action='store_true', help='Type of class-imbalance')
parser.add_argument('--val-iteration', type=int, default=1024,
                        help='Frequency for the evaluation')

# Hyperparameters for ReMixMatch
parser.add_argument('--mix_alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=1.5, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--w_rot', default=0.5, type=float)
parser.add_argument('--w_ent', default=0.5, type=float)

# Hyperparameters for CReST
parser.add_argument('--num_generation', default=6, type=int)
parser.add_argument('--dalign_t', default=0.5, type=float, help='t_min, 0.5 for FixMatch 0.8 for MixMatch')
parser.add_argument('--crest_alpha', default=3, type=float, help='3 for FixMatch 2 for MixMatch')
parser.add_argument('--no_scheduler', action='store_true', default=True, help='Type of class-imbalance')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
np.random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
cudnn.benchmark = False
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
    torch.backends.cudnn.deterministic = True

best_acc = 0  # best test accuracy
if args.dataset == 'cifar10':
    num_class = 10
elif args.dataset == 'cifar100':
    num_class = 100
else:
    raise NotImplementedError


def annotate_unlabeled_data(loader, model, use_cuda, num_classes):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Annotating unlabeled data...', max=len(loader))

    logits_scores = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs, _ = model(inputs)
            _, pred_label = torch.max(outputs, dim=1)
            logits_scores.append(outputs.cpu())

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

             # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                          'top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()

    return torch.cat(logits_scores, dim=0)


def get_split(x_train, y_train, x_unlab_test, y_unlab_test, class_im_ratio, num_classes, alpha=3, pseudo_label_list=None):
    if pseudo_label_list is not None:
        x_picked = []
        y_picked = []
        class_im_ratio = 1. / class_im_ratio

        mu = np.math.pow(class_im_ratio, 1 / (num_class - 1))
        for c in range(num_classes):
            num_picked = int(
                len(pseudo_label_list[c]) *
                np.math.pow(np.math.pow(mu, (num_class - 1) - c), 1 / alpha))
            idx_picked = pseudo_label_list[c][:num_picked]
            x_picked.append(x_unlab_test[idx_picked])
            y_picked.append(np.ones_like(y_unlab_test[idx_picked]) * c)
            print('class {} is added {} pseudo labels'.format(c, num_picked))
        x_picked.append(x_train)
        y_picked.append(y_train)
        x_train = np.concatenate(x_picked, axis=0)
        y_train = np.concatenate(y_picked, axis=0)
    else:
        print('not update')
    print('{} train set images in total'.format(len(x_train)))
    return x_train, y_train


def train_split(labels, n_labeled_per_class, n_unlabeled_per_class, num_classes, seed=0):
    np.random.seed(seed)
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        if seed != 0:
            np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        # note here train_labeled_idxs and train_unlabeled_idxs are disjoint
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])

    return train_labeled_idxs, train_unlabeled_idxs


class CIFAR_custom(data.Dataset):
    def __init__(self, x, y, transform=None, target_transform=None,):
        super(CIFAR_custom, self).__init__()
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(x) == len(y)
        assert x.shape[1] == 32
        assert x.shape[2] == 32
        assert x.shape[3] == 3

        self.data = x
        self.targets = y
        self.transform = transform
        self.target_transform = target_transform
        self.data = [Image.fromarray(img) for img in self.data]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)


class TransformTwice:
    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        out3 = self.transform2(inp)
        return out1, out2, out3


def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing imbalanced {args.dataset}')

    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, num_class, args.imb_ratio_l)
    U_SAMPLES_PER_CLASS = make_imb_data(args.ratio * args.num_max, num_class, args.imb_ratio_u)
    N_SAMPLES_PER_CLASS_T = torch.Tensor(N_SAMPLES_PER_CLASS)

    # Use the inferred distribution with labeled data
    num_samples_per_class = N_SAMPLES_PER_CLASS_T / N_SAMPLES_PER_CLASS_T.sum() * sum(U_SAMPLES_PER_CLASS)
    print(num_samples_per_class)
    target_disb = num_samples_per_class / num_samples_per_class.sum()


    # Main function
    pseudo_label_list = None
    if args.resume:
        tmp = args.resume.split('/')
        start_gen = int(tmp[-1][-9])
    else:
        start_gen = 0
    for gen_idx in range(start_gen, args.num_generation):

        cur = gen_idx / (args.num_generation - 1)
        current_dalign_t = (1.0 - cur) * 1.0 + cur * args.dalign_t

        if args.dataset == 'cifar10':
            base_dataset = torchvision.datasets.CIFAR10('/BS/databases00/cifar-10', train=True, download=False)
            test_set = torchvision.datasets.CIFAR10('/BS/databases00/cifar-10', train=False, download=False, transform=tfms_test_cifar10)
            weakDA = tfms_w_cifar10
            strongDA = tfms_s_cifar10
            valDA = tfms_test_cifar10
        elif args.dataset == 'cifar100':
            base_dataset = torchvision.datasets.CIFAR100('/BS/databases00/cifar-100', train=True, download=False)
            test_set = torchvision.datasets.CIFAR100('/BS/databases00/cifar-100', train=False, download=False, transform=tfms_test_cifar100)
            weakDA = tfms_w_cifar100
            strongDA = tfms_s_cifar100
            valDA = tfms_test_cifar100
        else:
            raise NotImplementedError

        train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.targets, N_SAMPLES_PER_CLASS,
                                                               U_SAMPLES_PER_CLASS, num_class, seed=args.manualSeed)

        x_train = base_dataset.data[train_labeled_idxs]
        y_train = np.array(base_dataset.targets)[train_labeled_idxs]
        x_unlab_test = base_dataset.data[train_unlabeled_idxs]
        y_unlab_test = np.array(base_dataset.targets)[train_unlabeled_idxs]
        x_unlab = np.concatenate([x_train, x_unlab_test], axis=0)
        y_unlab = np.concatenate([y_train, y_unlab_test], axis=0)
        x_train, y_train = get_split(x_train, y_train, x_unlab_test, y_unlab_test, args.imb_ratio_l,
                                     num_class, alpha=args.crest_alpha, pseudo_label_list=pseudo_label_list)

        train_labeled_set = CIFAR_custom(x_train, y_train, transform=strongDA)
        train_unlabeled_set = CIFAR_custom(x_unlab, y_unlab, transform=TransformTwice(weakDA, strongDA))
        unlabeled_anno_set = CIFAR_custom(x_unlab_test, y_unlab_test, transform=valDA)

        labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
        unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
        unlabeled_anno_loader = data.DataLoader(unlabeled_anno_set, batch_size=100, shuffle=False, drop_last=False)
        test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)

        # Model
        print("==> creating WRN-28-2")

        def create_model(ema=False):
            model = models.WRN(2, num_class)
            model = model.cuda()

            if ema:
                for param in model.parameters():
                    param.detach_()

            return model

        model = create_model()
        ema_model = create_model(ema=True)

        print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

        train_criterion = SemiLoss()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)

        if args.no_scheduler:
            scheduler = None
        else:
            total_steps = args.val_iteration * args.epochs
            scheduler = get_cosine_schedule_with_warmup(optimizer, 0, total_steps)
        # scheduler = get_cosine_schedule_with_warmup(optimizer, 0, total_steps)
        ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)
        start_epoch = 0

        title = 'remix-cifar-'
        if args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
            args.out = os.path.dirname(args.resume)
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            ema_model.load_state_dict(checkpoint['ema_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler'])
            logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
        else:
            logger = Logger(os.path.join(args.out, f'log_gen{gen_idx}.txt'), title=title)
            logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U', 'Total Acc.', 'Test Loss', 'Test Acc.', 'Test GM.'])

        test_accs = []
        test_gms = []

        # Default values for ReMixMatch and DARP
        emp_distb_u = torch.ones(num_class) / num_class
        for epoch in range(start_epoch, args.epochs):
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

            # Training part
            *train_info, emp_distb_u = train(labeled_trainloader, unlabeled_trainloader, model, optimizer, scheduler,
                                             ema_optimizer, train_criterion, epoch, use_cuda, target_disb, emp_distb_u,
                                             current_dalign_t)

            # Evaluation part
            test_loss, test_acc, test_cls, test_gm = validate(test_loader, ema_model, criterion, use_cuda, mode='Test Stats ')

            # Append logger file
            logger.append([*train_info, test_loss, test_acc, test_gm])

            # Save models
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler is not None else None,
            }, epoch + 1, filename=f'checkpoint_gen{gen_idx}.pth.tar')
            test_accs.append(test_acc)
            test_gms.append(test_gm)

        logger.close()

        # Print the final results
        print('Mean bAcc:')
        print(np.mean(test_accs[-20:]))

        print('Mean GM:')
        print(np.mean(test_gms[-20:]))

        print('Name of saved folder:')
        print(args.out)

        with torch.no_grad():
            logits_scores = annotate_unlabeled_data(unlabeled_anno_loader, ema_model, use_cuda, num_class)

            y_pred = torch.argmax(logits_scores, dim=1)  # [len(whole data set), ]
            y_score = torch.max(logits_scores, dim=1)[0]

            pseudo_label_list = []  # list of np.arr, each of which contains idx of data that has label class_idx (sorted)
            for class_idx in range(num_class):
                idx_gather = torch.nonzero(y_pred == class_idx).view(-1)
                score_gather = y_score[idx_gather]
                _, order = score_gather.sort(descending=True)
                idx_gather = idx_gather[order]
                pseudo_label_list.append(idx_gather.numpy())


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, scheduler, ema_optimizer, criterion, epoch, use_cuda,
          target_disb, emp_distb_u, current_dalign_t):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    losses_r = AverageMeter()
    losses_e = AverageMeter()
    total_c = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x, _ = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, _ = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2, inputs_u3), gt_targets_u, idx_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2, inputs_u3), gt_targets_u, idx_u = unlabeled_train_iter.next()

        # Measure data loading time
        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, num_class).scatter_(1, targets_x.view(-1, 1), 1)
        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u, inputs_u2, inputs_u3 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda()

        # Rotate images
        temp = []
        targets_r = torch.randint(0, 4, (inputs_u2.size(0),)).long()
        for i in range(inputs_u2.size(0)):
            inputs_rot = torch.rot90(inputs_u2[i], targets_r[i], [1, 2]).reshape(1, 3, 32, 32)
            temp.append(inputs_rot)
        inputs_r = torch.cat(temp, 0)
        targets_r = torch.zeros(batch_size, 4).scatter_(1, targets_r.view(-1, 1), 1)
        inputs_r, targets_r = inputs_r.cuda(), targets_r.cuda(non_blocking=True)

        # Generate the pseudo labels
        with torch.no_grad():
            outputs_u, _ = model(inputs_u)
            p = torch.softmax(outputs_u, dim=1)

            # Tracking the empirical distribution on the unlabeled samples (ReMixMatch)
            real_batch_idx = batch_idx + epoch * args.val_iteration
            if real_batch_idx == 0:
                emp_distb_u = p.mean(0, keepdim=True)
            elif real_batch_idx // 128 == 0:
                emp_distb_u = torch.cat([emp_distb_u, p.mean(0, keepdim=True)], 0)
            else:
                emp_distb_u = emp_distb_u[-127:]
                emp_distb_u = torch.cat([emp_distb_u, p.mean(0, keepdim=True)], 0)

            t_scaled_target_disb = target_disb ** current_dalign_t
            t_scaled_target_disb /= t_scaled_target_disb.sum()
            pa = p * (t_scaled_target_disb.cuda() + 1e-6) / (emp_distb_u.mean(0).cuda() + 1e-6)
            p = pa / pa.sum(dim=1, keepdim=True)

            # Temperature scailing
            pt = p ** (1 / args.T)
            targets_u = (pt / pt.sum(dim=1, keepdim=True)).detach()

            max_p, p_hat = torch.max(targets_u, dim=1)
            total_acc = p_hat.cpu().eq(gt_targets_u).float().view(-1)
            total_c.update(total_acc.mean(0).item())

        # Mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2, inputs_u3], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u, targets_u], dim=0)

        l = np.random.beta(args.mix_alpha, args.mix_alpha)
        l = max(l, 1 - l)
        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])[0]]
        for input in mixed_input[1:]:
            logits.append(model(input)[0])

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:],
                              epoch + batch_idx / args.val_iteration)
        _, logits_r = model(inputs_r)
        Lr = -1 * torch.mean(torch.sum(F.log_softmax(logits_r, dim=1) * targets_r, dim=1))

        # Entropy minimization for unlabeled samples (strong augmented)
        outputs_u2, _ = model(inputs_u2)
        Le = -1 * torch.mean(torch.sum(F.log_softmax(outputs_u2, dim=1) * targets_u, dim=1))
        loss = Lx + w * Lu + args.w_rot * Lr + args.w_ent * Le * linear_rampup(epoch + batch_idx / args.val_iteration)

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        losses_r.update(Lr.item(), inputs_x.size(0))
        losses_e.update(Le.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                     'Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | Loss_r: {loss_r:.4f} | ' \
                     'Loss_e: {loss_e:.4f} | Total_acc: {total_c:.4f}'.format(
            batch=batch_idx + 1,
            size=args.val_iteration,
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            loss_x=losses_x.avg,
            loss_u=losses_u.avg,
            loss_r=losses_r.avg,
            loss_e=losses_e.avg,
            total_c=total_c.avg,
        )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg, total_c.avg, emp_distb_u)


def validate(valloader, model, criterion, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    classwise_correct = torch.zeros(num_class)
    classwise_num = torch.zeros(num_class)
    section_acc = torch.zeros(3)

    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)
            y_true.extend(targets.tolist())

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # classwise prediction
            pred_label = outputs.max(1)[1]
            y_pred.extend(pred_label.tolist())
            pred_mask = (targets == pred_label).float()
            for i in range(num_class):
                class_mask = (targets == i).float()

                classwise_correct[i] += (class_mask * pred_mask).sum()
                classwise_num[i] += class_mask.sum()

             # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                          'Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()

    # Major, Neutral, Minor
    section_num = int(num_class / 3)
    classwise_acc = (classwise_correct / classwise_num)
    section_acc[0] = classwise_acc[:section_num].mean()
    section_acc[2] = classwise_acc[-1 * section_num:].mean()
    section_acc[1] = classwise_acc[section_num:-1 * section_num].mean()
    GM = 1
    for i in range(num_class):
        if classwise_acc[i] == 0:
            # To prevent the N/A values, we set the minimum value as 0.001
            GM *= (1/(100 * num_class)) ** (1/num_class)
        else:
            GM *= (classwise_acc[i]) ** (1/num_class)

    return (losses.avg, top1.avg, section_acc.numpy(), GM)


def make_imb_data(max_num, class_num, gamma):
    print(gamma)
    mu = np.power(1/abs(gamma), 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / abs(gamma)))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    if gamma < 0:
        class_num_list = class_num_list[::-1]
    print(class_num_list)
    return list(class_num_list)


def save_checkpoint(state, epoch, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    # if epoch % 100 == 0:
    #     shutil.copyfile(filepath, os.path.join(checkpoint, 'model_' + str(epoch) + '.pth.tar'))


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=7. / 16., last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))  # this is correct

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1))

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param.mul_(1 - self.wd)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


if __name__ == '__main__':
    main()