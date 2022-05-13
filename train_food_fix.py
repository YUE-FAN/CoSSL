"""
The script is based on aves_fix.py

This script trains a resnet50 (from scratch) on Food-101.
I use RandAugment from https://github.com/kekmodel/FixMatch-pytorch/blob/master/dataset/randaugment.py
Hypers are from A Realistic Evaluation of Semi-Supervised Learning for Fine-Grained Classification.

debug:
todo: I changed the model.cuda(0) into model.cuda()
"""
from __future__ import print_function

import argparse
import os
import shutil
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
from torch.optim.lr_scheduler import LambdaLR
import torchvision

import dataset.fix_food as dataset
from utils import make_imb_data, save_checkpoint, FixMatch_Loss
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p

parser = argparse.ArgumentParser(description='PyTorch ReMixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.04, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--wd', default=0.0003, type=float,
                    help='weight decay')
parser.add_argument('--warmup_epochs', default=5, type=int, help='number of warmup epochs to run')
parser.add_argument('--decay_epochs', default=30, type=int, help='number of decay_epochs')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--out', default='result',
                        help='Directory to output the result')
# Method options
parser.add_argument('--num_max', type=int, default=250,
                        help='Number of samples in the maximal class')
parser.add_argument('--ratio', type=float, default=2.0,
                        help='Relative size between labeled and unlabeled data')
parser.add_argument('--imb_ratio_l', type=int, default=50,
                        help='Imbalance ratio for labeled data')
parser.add_argument('--imb_ratio_u', type=int, default=50,
                        help='Imbalance ratio for unlabeled data')
# parser.add_argument('--val-iteration', type=int, default=500,
#                         help='Frequency for the evaluation')
# Hyperparameters for FixMatch
parser.add_argument('--tau', default=0.9, type=float, help='hyper-parameter for pseudo-label of FixMatch')  # todo: not sure
parser.add_argument('--ema-decay', default=0.999, type=float)
parser.add_argument('--mu', default=2, type=float, help='unlabeled bs = mu * bs')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
# parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
# state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
# os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
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
num_class = 101


def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print('==> Preparing imbalanced Food-101')

    N_SAMPLES_PER_CLASS = make_imb_data(args.num_max, num_class, args.imb_ratio_l)
    U_SAMPLES_PER_CLASS = make_imb_data(args.ratio * args.num_max, num_class, args.imb_ratio_u)
    N_SAMPLES_PER_CLASS_T = torch.Tensor(N_SAMPLES_PER_CLASS)

    train_labeled_set, train_unlabeled_set, test_set = dataset.get_food101('/BS/yfan/nobackup/food-101/',
                                                                           N_SAMPLES_PER_CLASS, U_SAMPLES_PER_CLASS,
                                                                           seed=args.manualSeed)

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                          drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.mu * args.batch_size, shuffle=True,
                                            num_workers=8, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)
    args.val_iteration = len(unlabeled_trainloader)
    print(len(train_unlabeled_set), args.mu * args.batch_size, args.val_iteration)

    # Model
    print("==> creating ResNet50")

    def create_model(ema=False):
        model = torchvision.models.resnet50(num_classes=num_class)
        model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    train_criterion = FixMatch_Loss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=True)
    ema_optimizer = WeightEMA(model, ema_model, args.lr, alpha=args.ema_decay)
    warmup_steps = args.warmup_epochs * args.val_iteration  # int
    scheduler = stepLR_with_warmup(optimizer, warmup_steps, args.val_iteration)
    # total_steps = args.val_iteration * args.epochs
    # scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    start_epoch = 0

    # Resume
    title = 'fix-food-101'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        # args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U', 'Mask', 'Total Acc.', 'Used Acc.',
                          'Test Loss', 'Test Acc.', 'Test GM.'])

    test_accs = []
    test_gms = []

    model = nn.DataParallel(model)

    # Main function
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))

        target_disb = N_SAMPLES_PER_CLASS_T * len(train_unlabeled_set) / sum(N_SAMPLES_PER_CLASS)
        # In case of FixMatch, labeled data is utilized as unlabeled data once again.
        target_disb += N_SAMPLES_PER_CLASS_T    # (10,) how many unlabeled samples per class

        # Training part
        *train_info, = train(labeled_trainloader,
                             unlabeled_trainloader,
                             model, optimizer,
                             ema_optimizer,
                             scheduler,
                             train_criterion,
                             epoch, use_cuda)

        # Evaluation part
        test_loss, test_acc, test_cls, test_gm = validate(test_loader, ema_model, criterion, use_cuda, mode='Test Stats ')

        # Append logger file
        logger.append([*train_info, test_loss, test_acc, test_gm])

        # Save models
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.module.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, epoch, args.out)

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


def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, scheduler, criterion, epoch, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    mask_prob = AverageMeter()
    total_c = AverageMeter()
    used_c = AverageMeter()
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
            (inputs_u, inputs_u2), gt_targets_u, idx_u = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), gt_targets_u, idx_u = unlabeled_train_iter.next()

        # Measure data loading time
        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, num_class).scatter_(1, targets_x.view(-1,1), 1)
        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(0), targets_x.cuda(0)
            inputs_u, inputs_u2 = inputs_u.cuda(0), inputs_u2.cuda(0)

        # Generate the pseudo labels
        with torch.no_grad():
            # Generate the pseudo labels by aggregation and sharpening
            outputs_u_list = []
            for i in range(int(args.mu)):
                tmp = model(inputs_u[batch_size * i: batch_size * (i + 1), :, :, :])
                outputs_u_list.append(tmp)
            outputs_u = torch.cat(outputs_u_list, dim=0)
            # outputs_u = model(inputs_u)
            targets_u = torch.softmax(outputs_u, dim=1)

            max_p, p_hat = torch.max(targets_u, dim=1)
            select_mask = max_p.ge(args.tau).float()

            total_acc = p_hat.cpu().eq(gt_targets_u).float().view(-1)
            if select_mask.sum() != 0:
                used_c.update(total_acc[select_mask != 0].mean(0).item(), select_mask.sum())
            mask_prob.update(select_mask.mean().item())
            total_c.update(total_acc.mean(0).item())

            p_hat = torch.zeros(batch_size * args.mu, num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)
            # select_mask = torch.cat([select_mask, select_mask], 0)

        logits_x = model(inputs_x)
        logits_u_list = []
        for i in range(int(args.mu)):
            tmp = model(inputs_u2[batch_size * i: batch_size * (i + 1), :, :, :])
            logits_u_list.append(tmp)
        logits_u = torch.cat(logits_u_list, dim=0)

        Lx, Lu = criterion(logits_x, targets_x, logits_u, p_hat, select_mask)
        loss = Lx + Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))

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
                      'Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | Mask: {mask:.4f}| ' \
                      'Use_acc: {used_acc:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_prob.avg,
                    used_acc=used_c.avg,
                    )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg, mask_prob.avg, total_c.avg, used_c.avg)

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

    classwise_correct = torch.zeros(num_class).cuda()
    classwise_num = torch.zeros(num_class).cuda()
    section_acc = torch.zeros(3).cuda()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)
            y_true.extend(targets.tolist())

            if use_cuda:
                inputs, targets = inputs.cuda(0), targets.cuda(0)
            # compute output
            outputs = model(inputs)
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
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
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

    return (losses.avg, top1.avg, section_acc.cpu().numpy(), GM)


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7. / 16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))  # this is correct

    return LambdaLR(optimizer, _lr_lambda, last_epoch)


def stepLR_with_warmup(optimizer, num_warmup_steps, per_epoch_steps):

    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(num_warmup_steps)
        else:
            return args.gamma ** ((current_step-num_warmup_steps) // (args.decay_epochs * per_epoch_steps))

    return LambdaLR(optimizer, _lr_lambda, -1)


class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # # customized weight decay
            # param.mul_(1 - self.wd)


if __name__ == '__main__':
    main()
