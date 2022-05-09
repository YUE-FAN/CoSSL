"""
This script is based on small_imagenet127_fix_teacher_head.py

"""

from __future__ import print_function

import argparse
import os
import json

import copy
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import time
from PIL import Image
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
from torchvision.transforms.functional import normalize
from torch.utils.data.sampler import WeightedRandomSampler, BatchSampler
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

import models.resnet as models
from dataset.fix_small_imagenet127 import get_small_imagenet

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from scipy import optimize

parser = argparse.ArgumentParser(description='PyTorch ReMixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=64, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_tfe', default=0.002, type=float)
parser.add_argument('--wd_tfe', default=5e-4, type=float)
parser.add_argument('--warm_tfe', default=10, type=int)
# Checkpoints
parser.add_argument('--resume', default=' ', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--out', default='result', help='Directory to output the result')
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

# Method options
parser.add_argument('--labeled_percent', type=float, default=0.1, help='by default we take 10% labeled data')
parser.add_argument('--img_size', type=int, default=32, help='ImageNet127_32 or ImageNet127_64')
parser.add_argument('--val-iteration', type=int, default=500, help='Frequency for the evaluation')

# Hyperparameters for FixMatch
parser.add_argument('--tau', default=0.95, type=float, help='hyper-parameter for pseudo-label of FixMatch')
parser.add_argument('--ema-decay', default=0.999, type=float)

# Hyperparameters for DARP
parser.add_argument('--warm', type=int, default=200,  help='Number of warm up epoch for DARP')
parser.add_argument('--alpha', default=2.0, type=float, help='hyperparameter for removing noisy entries')
parser.add_argument('--darp', action='store_true', help='Applying DARP')
parser.add_argument('--est', action='store_true', help='Using estimated distribution for unlabeled dataset')
parser.add_argument('--iter_T', type=int, default=10, help='Number of iteration (T) for DARP')
parser.add_argument('--num_iter', type=int, default=10, help='Scheduling for updating pseudo-labels')

# Hyperparameters for cRT
parser.add_argument('--crt_u_ratio', default=1, type=int)
# parser.add_argument('--crt_mix_alpha', default=0.75, type=float)
parser.add_argument('--crt_weakDA', action='store_true', help='use weakDA during crt')
# parser.add_argument('--maxlam', action='store_true', help='use max(lam, 1-lam)')
parser.add_argument('--garbage_percent', type=float, default=0., help='percentage of garbage images for each class')
parser.add_argument('--ema_encoder', action='store_true', help='teacher ema')
parser.add_argument('--ema_head', action='store_true', help='teacher ema')
parser.add_argument('--similar', action='store_true', help='select the most similar features')
parser.add_argument('--max_lam', default=0.8, type=float)

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
num_class = 127


def get_weighted_sampler(target_sample_rate, num_sample_per_class, target):
    assert len(num_sample_per_class) == len(np.unique(target))

    sample_weights = target_sample_rate / num_sample_per_class  # this is the key line!!!
    print(sample_weights)

    # assign each sample a weight by sampling rate
    samples_weight = np.array([sample_weights[t] for t in target])

    return WeightedRandomSampler(samples_weight, len(samples_weight), True)


class merge_two_datasets(data.Dataset):
    def __init__(self, data1, data2, targets1, targets2,
                 transform=None, target_transform=None):
        self.data = copy.deepcopy(np.concatenate([data1, data2], axis=0))
        self.targets = copy.deepcopy(np.concatenate([targets1, targets2], axis=0))
        assert len(self.data) == len(self.targets)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)


def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing imbalanced ImageNet127-{args.img_size}')

    img_size2path = {32: '/BS/yfan/nobackup/ImageNet127_32', 64: '/BS/yfan/nobackup/ImageNet127_64'}
    tmp = get_small_imagenet(img_size2path[args.img_size], args.img_size, labeled_percent=args.labeled_percent,
                             seed=args.manualSeed, return_strong_labeled_set=True)
    target_disb, train_labeled_set, train_unlabeled_set, test_set, train_strong = tmp

    N_SAMPLES_PER_CLASS = [0 for _ in range(num_class)]
    for l in train_labeled_set.targets:
        N_SAMPLES_PER_CLASS[l] += 1
    print(N_SAMPLES_PER_CLASS)

    if args.crt_weakDA:
        crt_labeled_set = copy.deepcopy(train_labeled_set)
    else:
        crt_labeled_set = copy.deepcopy(train_strong)

    crt_full_set = merge_two_datasets(crt_labeled_set.data, train_unlabeled_set.data, crt_labeled_set.targets,
                                      train_unlabeled_set.targets, transform=crt_labeled_set.transform)

    labeled_trainloader = data.DataLoader(train_labeled_set, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                          drop_last=True)
    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                            drop_last=True)
    crt_full_loader = data.DataLoader(crt_full_set, batch_size=args.crt_u_ratio * args.batch_size, shuffle=True, num_workers=8,
                                            drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=4)

    # Model
    print("==> creating WRN-28-2")

    def create_model(ema=False, clf_bias=True):
        model = models.ResNet50(num_classes=num_class, rotation=True, classifier_bias=clf_bias)
        model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model(clf_bias=True)
    ema_model = create_model(ema=True, clf_bias=True)
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)
    start_epoch = 0

    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    ema_model.load_state_dict(checkpoint['ema_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    # todo: continue training with weight decay??
    for group in optimizer.param_groups:
        group['weight_decay'] = 0.02 * args.lr
    logger = Logger(os.path.join(args.out, 'log.txt'), title='fix-cifar')
    logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U', 'Train Loss Teacher', 'Mask', 'Total Acc.', 'Used Acc.', 'Teacher Acc.',
                      'Test Loss', 'Test Acc. online Teacher', 'Test Acc. EMA Teacher', 'Test Acc. EMA', 'Test Acc. online'])
    _, test_acc_online, *_ = validate(test_loader, model, nn.CrossEntropyLoss(), use_cuda, mode='Test ckpt')
    _, test_acc_ema, *_ = validate(test_loader, ema_model, nn.CrossEntropyLoss(), use_cuda, mode='Test ckpt')
    logger.append([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, test_acc_ema, test_acc_online])

    # todo: add a teacher head
    teacher_head = nn.Linear(model.output.in_features, num_class, bias=True).cuda()
    ema_teacher = nn.Linear(model.output.in_features, num_class, bias=True).cuda()
    # nn.init.xavier_normal_(teacher_head.weight)
    # nn.init.constant_(teacher_head.bias, 0)
    for param in ema_teacher.parameters():
        param.detach_()
    wd_params, non_wd_params = [], []
    for name, param in teacher_head.named_parameters():
        if 'bn' in name or 'bias' in name:
            non_wd_params.append(param)
        else:
            wd_params.append(param)

    param_list = [{'params': wd_params, 'weight_decay': args.wd_tfe}, {'params': non_wd_params, 'weight_decay': 0}]
    # teacher_optimizer = optim.SGD(param_list, lr=lr, momentum=0.9, nesterov=True)
    teacher_optimizer = optim.Adam(param_list, lr=args.lr_tfe)
    ema_teacher_optimizer = WeightEMA(teacher_head, ema_teacher, alpha=args.ema_decay)

    # do cRT with feature mixUp once
    init_teacher, init_ema_teacher = cRT_feat_mixup(copy.deepcopy(ema_model), crt_labeled_set, crt_full_set,
                                                    test_set, N_SAMPLES_PER_CLASS, num_class, use_cuda)
    teacher_head.weight.data.copy_(init_teacher.output.weight.data)
    teacher_head.bias.data.copy_(init_teacher.output.bias.data)
    ema_teacher.weight.data.copy_(init_ema_teacher.output.weight.data)
    ema_teacher.bias.data.copy_(init_ema_teacher.output.bias.data)



    test_accs = []

    # Default values for ReMixMatch and DARP
    emp_distb_u = torch.ones(num_class) / num_class
    pseudo_orig = torch.ones(len(train_unlabeled_set.targets), num_class) / num_class
    pseudo_refine = torch.ones(len(train_unlabeled_set.targets), num_class) / num_class


    # Main function
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        # todo: create a copy of the labeled data???
        class_balanced_disb = torch.Tensor(make_imb_data(30000, num_class, 1))
        class_balanced_disb = class_balanced_disb / class_balanced_disb.sum()
        sampler_x = get_weighted_sampler(class_balanced_disb, torch.Tensor(N_SAMPLES_PER_CLASS), crt_labeled_set.targets)
        batch_sampler_x = BatchSampler(sampler_x, batch_size=args.batch_size, drop_last=True)
        crt_labeled_loader = data.DataLoader(crt_labeled_set, batch_sampler=batch_sampler_x, num_workers=8)

        # # Use the estimated distribution of unlabeled data
        # if args.est:
        #     est_name = './estimation/cifar10@N_1500_r_{}_{}_estim.npy'.format(args.imb_ratio_l, args.imb_ratio_u)
        #     est_disb = np.load(est_name)
        #     target_disb = sum(U_SAMPLES_PER_CLASS) * torch.Tensor(est_disb) / np.sum(est_disb)
        # # Use the inferred distribution with labeled data
        # else:
        #     target_disb = N_SAMPLES_PER_CLASS_T * sum(U_SAMPLES_PER_CLASS) / sum(N_SAMPLES_PER_CLASS)
        # # In case of FixMatch, labeled data is utilized as unlabeled data once again.
        # target_disb += N_SAMPLES_PER_CLASS_T    # (10,) how many unlabeled samples per class

        # Training part
        *train_info, emp_distb_u, pseudo_orig, pseudo_refine = train(labeled_trainloader, unlabeled_trainloader, model, ema_model, optimizer, ema_optimizer,
                                                                     crt_labeled_loader, crt_full_loader, teacher_head, ema_teacher, teacher_optimizer, ema_teacher_optimizer,
                                                                     train_criterion, epoch, use_cuda,
                                                                     N_SAMPLES_PER_CLASS, target_disb,
                                                                     emp_distb_u, pseudo_orig, pseudo_refine)

        # Evaluation part
        _, test_acc_online, *_ = validate(test_loader, model, criterion, use_cuda,
                                          mode='Test online model', log_name='per_class_acc_online_model.json')
        test_loss, test_acc_ema, *_ = validate(test_loader, ema_model, criterion, use_cuda,
                                               mode='Test ema model', log_name='per_class_acc_ema_model.json')
        _, test_acc_online_teacher, *_ = validate_teacher(test_loader, ema_model, teacher_head, criterion, use_cuda,
                                                   mode='Test online teacher', log_name='per_class_acc_online_teacher.json')
        _, test_acc_ema_teacher, *_ = validate_teacher(test_loader, ema_model, ema_teacher, criterion, use_cuda,
                                                   mode='Test ema teacher', log_name='per_class_acc_ema_teacher.json')

        # Append logger file
        logger.append([*train_info, test_loss, test_acc_online_teacher, test_acc_ema_teacher, test_acc_ema, test_acc_online])

        # Save models
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'ema_state_dict': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'teacher_optimizer': teacher_optimizer.state_dict(),
            'teacher_head': teacher_head.state_dict(),
            'ema_teacher': ema_teacher.state_dict(),
        }, epoch + 1)
        test_accs.append(test_acc_ema)

    # teacher = cRT_feat_mixup(copy.deepcopy(model), train_labeled_set, train_unlabeled_set_anno, test_set,
    #                          N_SAMPLES_PER_CLASS, num_class, use_cuda)
    # _, test_acc_crt, *_ = validate(test_loader, teacher, nn.CrossEntropyLoss(), use_cuda, mode='eval teacher')
    # logger.append([-1, -1, -1, -1, -1, -1, -1, -1, test_acc_crt, test_acc_crt])
    logger.close()

    # Print the final results
    print('Mean bAcc:')
    print(np.mean(test_accs[-20:]))

    print('Name of saved folder:')
    print(args.out)



def cRT_feat_mixup(model, train_labeled_set, train_unlabeled_set, test_set, N_SAMPLES_PER_CLASS, num_class, use_cuda):

    # define hypers for cRT
    val_iteration = args.val_iteration
    epochs = args.warm_tfe
    lr = args.lr_tfe
    ema_decay = 0.999
    weight_decay = args.wd_tfe
    batch_size = args.batch_size

    # construct dataloaders for cRT
    class_balanced_disb = torch.Tensor(make_imb_data(30000, num_class, 1))
    class_balanced_disb = class_balanced_disb / class_balanced_disb.sum()
    sampler_x = get_weighted_sampler(class_balanced_disb, torch.Tensor(N_SAMPLES_PER_CLASS), train_labeled_set.targets)
    batch_sampler_x = BatchSampler(sampler_x, batch_size=batch_size, drop_last=True)
    labeled_trainloader = data.DataLoader(train_labeled_set, batch_sampler=batch_sampler_x, num_workers=8)

    unlabeled_trainloader = data.DataLoader(train_unlabeled_set, batch_size=batch_size,
                                            shuffle=False, num_workers=8, drop_last=False)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    mixup_model = get_imprint_model(copy.deepcopy(model), train_labeled_set, test_set, num_class)

    # fix the feature extractor and reinitialize the classifier
    for param in model.parameters():
        param.requires_grad = False
    model.output.reset_parameters()
    for param in model.output.parameters():
        param.requires_grad = True

    ema_model = copy.deepcopy(model)
    for param in ema_model.parameters():
        param.detach_()
    ema_optimizer = WeightEMA(model, ema_model, alpha=ema_decay)

    wd_params, non_wd_params = [], []
    for name, param in model.output.named_parameters():
        if 'bn' in name or 'bias' in name:
            non_wd_params.append(param)  # bn.weight, bn.bias and classifier.bias, conv2d.bias
        else:
            wd_params.append(param)
    param_list = [{'params': wd_params, 'weight_decay': weight_decay}, {'params': non_wd_params, 'weight_decay': 0}]
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.output.parameters()) / 1000000.0))

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(param_list, lr=lr, momentum=0.9, nesterov=True)
    optimizer = optim.Adam(param_list, lr=lr)
    # total_steps = val_iteration * epochs
    # scheduler = get_cosine_schedule_with_warmup(optimizer, 0, total_steps)

    # Main function
    for epoch in range(epochs):
        print('\ncRT: Epoch: [%d | %d] LR: %f' % (epoch + 1, epochs, optimizer.param_groups[0]['lr']))

        cRT_train(labeled_trainloader, unlabeled_trainloader, model, optimizer, None, ema_optimizer,
                  mixup_model, N_SAMPLES_PER_CLASS, val_iteration, use_cuda)
        validate(test_loader, ema_model, criterion, use_cuda, mode='cRT: test EMA model')

    return model, ema_model


def get_imprint_model(model, labeled_set, test_set, num_classes):
    """
    better pass a copy of an online model to model parameter
    """
    model = model.cuda()
    model.eval()

    testloader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=0, drop_last=False)
    labeledloader = torch.utils.data.DataLoader(labeled_set, batch_size=100, shuffle=False, num_workers=0, drop_last=False)

    with torch.no_grad():
        bar = Bar('Processing imprinting...', max=len(labeledloader))

        for batch_idx, (inputs, targets, _) in enumerate(labeledloader):
            inputs = inputs.cuda()
            _, _, features = model(inputs, True)
            output = features.squeeze()   # Note: a flatten is needed here

            if batch_idx == 0:
                output_stack = output.cpu()
                target_stack = targets
            else:
                output_stack = torch.cat((output_stack, output.cpu()), 0)
                target_stack = torch.cat((target_stack, targets), 0)

            bar.suffix = '({batch}/{size}'.format(batch=batch_idx + 1, size=len(labeledloader))
            bar.next()
        bar.finish()
    new_weight = torch.zeros(num_classes, model.output.in_features)
    for i in range(num_classes):
        tmp = output_stack[target_stack == i].mean(0)
        new_weight[i] = tmp / tmp.norm(p=2)
    model.output = torch.nn.Linear(model.output.in_features, num_classes, bias=False).cuda()
    model.output.weight.data = new_weight.cuda()
    validate(testloader, model, nn.CrossEntropyLoss(), True, mode='Imprint Test')

    model.eval()
    return model


def cRT_train(labeled_loader, unlabeled_loader, model, optimizer, scheduler, ema_optimizer,
              mixup_model, num_samples_per_class, val_iteration, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    train_acc = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=val_iteration)
    labeled_train_iter = iter(labeled_loader)
    unlabeled_train_iter = iter(unlabeled_loader)

    model.eval()
    mixup_model.eval()

    mixup_prob = [(max(num_samples_per_class) - i) / max(num_samples_per_class) for i in num_samples_per_class]

    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x, _ = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_loader)
            inputs_x, targets_x, _ = labeled_train_iter.next()

        try:
            input_u, targets_u, _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_loader)
            input_u, targets_u, _ = unlabeled_train_iter.next()

        # Measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)  # targets are one-hot
            input_u = input_u.cuda()

        with torch.no_grad():
            _, _, crt_feat_x = mixup_model(inputs_x, return_feature=True)
            crt_feat_x = crt_feat_x.squeeze()
            _, _, crt_feat_u = mixup_model(input_u, return_feature=True)
            crt_feat_u = crt_feat_u.squeeze()

            new_feat_list = []
            new_target_list = []
            for x, label_x, u in zip(crt_feat_x, targets_x, crt_feat_u[:len(targets_x)]):
                if random.random() < mixup_prob[label_x]:
                    # lam = np.random.beta(args.crt_mix_alpha, args.crt_mix_alpha, size=1)
                    lam = np.random.uniform(args.max_lam, 1., size=1)
                    lam = torch.FloatTensor(lam).cuda()
                    # if args.maxlam:
                    #     lam = max(lam, 1 - lam)

                    new_feat = lam * x + (1 - lam) * u
                    new_target = label_x
                else:
                    new_feat = x
                    new_target = label_x
                new_feat_list.append(new_feat)
                new_target_list.append(new_target)
            new_feat_tensor = torch.stack(new_feat_list, dim=0)  # [64, 128]
            new_target_tensor = torch.stack(new_target_list, dim=0)  # [64,]

        logits = model.output(new_feat_tensor)
        # loss = -torch.mean(torch.sum(F.log_softmax(teacher_logits, dim=1) * new_target_tensor, dim=1))
        loss = F.cross_entropy(logits, new_target_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()
        if scheduler is not None:
            scheduler.step()

        # record loss
        acc = (torch.argmax(logits, dim=1) == new_target_tensor).float().mean()
        losses.update(loss.item(), inputs_x.size(0))
        train_acc.update(acc.item(), inputs_x.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                      'Loss: {loss:.4f} | Train_Acc: {train_acc:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    train_acc=train_acc.avg,
                    )
        bar.next()
    bar.finish()

    return (losses.avg, train_acc.avg)


# def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=7. / 16., last_epoch=-1):
#     def _lr_lambda(current_step):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
#         no_progress = float(current_step - num_warmup_steps) / \
#                       float(max(1, num_training_steps - num_warmup_steps))
#         return max(0., math.cos(math.pi * num_cycles * no_progress))  # this is correct
#
#     return LambdaLR(optimizer, _lr_lambda, last_epoch)


def train(labeled_trainloader, unlabeled_trainloader, model, ema_model, optimizer, ema_optimizer,
          crt_labeled_loader, crt_full_loader, teacher_head, ema_teacher, teacher_optimizer, ema_teacher_optimizer,
          criterion, epoch, use_cuda, num_labeled_data_per_class, target_disb, emp_distb_u, pseudo_orig, pseudo_refine):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    losses_teacher = AverageMeter()
    mask_prob = AverageMeter()
    total_c = AverageMeter()
    used_c = AverageMeter()
    teacher_acc = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)
    crt_labeled_iter = iter(crt_labeled_loader)
    crt_full_iter = iter(crt_full_loader)

    model.train()
    ema_model.eval()  # todo: maybe this is the reason why teacher gets worse and worse train acc?

    mixup_prob = [(max(num_labeled_data_per_class) - i) / max(num_labeled_data_per_class) for i in num_labeled_data_per_class]
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

        try:
            crt_input_x, crt_targets_x, _ = crt_labeled_iter.next()
        except:
            crt_labeled_iter = iter(crt_labeled_loader)
            crt_input_x, crt_targets_x, _ = crt_labeled_iter.next()

        try:
            crt_input_u, crt_targets_u, _ = crt_full_iter.next()
        except:
            crt_full_iter = iter(crt_full_loader)
            crt_input_u, crt_targets_u, _ = crt_full_iter.next()

        # Measure data loading time
        data_time.update(time.time() - end)
        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, num_class).scatter_(1, targets_x.view(-1, 1), 1)
        crt_targets_x = torch.zeros(batch_size, num_class).scatter_(1, crt_targets_x.view(-1, 1), 1)
        # crt_targets_u = torch.zeros(batch_size, num_class).scatter_(1, crt_targets_u.view(-1, 1), 1)
        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u, inputs_u2, inputs_u3 = inputs_u.cuda(), inputs_u2.cuda(), inputs_u3.cuda()
            crt_input_x, crt_input_u, crt_targets_x = crt_input_x.cuda(), crt_input_u.cuda(), crt_targets_x.cuda()

        # Generate the pseudo labels
        with torch.no_grad():
            # Generate the pseudo labels by ema_model and ema_teacher
            if args.ema_encoder:
                _, _, feature_u = ema_model(inputs_u, return_feature=True)
            else:
                _, _, feature_u = model(inputs_u, return_feature=True)
            if args.ema_head:
                outputs_u = ema_teacher(feature_u.squeeze())
            else:
                outputs_u = teacher_head(feature_u.squeeze())
            targets_u = torch.softmax(outputs_u, dim=1)

            # Update the saved predictions with current one
            pseudo_orig[idx_u, :] = targets_u.data.cpu()
            pseudo_orig_backup = pseudo_orig.clone()

            # Applying DARP
            if args.darp and epoch > args.warm:
                if batch_idx % args.num_iter == 0:
                    # Iterative normalization
                    targets_u, weights_u = estimate_pseudo(target_disb, pseudo_orig)
                    scale_term = targets_u * weights_u.reshape(1, -1)
                    pseudo_orig = (pseudo_orig * scale_term + 1e-6) \
                                      / (pseudo_orig * scale_term + 1e-6).sum(dim=1, keepdim=True)

                    opt_res = opt_solver(pseudo_orig, target_disb)

                    # Updated pseudo-labels are saved
                    pseudo_refine = opt_res

                    # Select
                    targets_u = opt_res[idx_u].detach().cuda()
                    pseudo_orig = pseudo_orig_backup
                else:
                    # Using previously saved pseudo-labels
                    targets_u = pseudo_refine[idx_u].cuda()

            # todo: I include the following code into torch.no_grad()
            max_p, p_hat = torch.max(targets_u, dim=1)
            select_mask = max_p.ge(args.tau).float()
            total_acc = p_hat.cpu().eq(gt_targets_u).float().view(-1)
            if select_mask.sum() != 0:
                used_c.update(total_acc[select_mask != 0].mean(0).item(), select_mask.sum())
            mask_prob.update(select_mask.mean().item())
            total_c.update(total_acc.mean(0).item())

            # max_p, p_hat = torch.max(teacher_targets, dim=1)
            # select_mask = max_p.ge(args.tau).float()
            # total_acc = p_hat.cpu().eq(gt_targets_u).float().view(-1)
            # if select_mask.sum() != 0:
            #     used_c_teacher.update(total_acc[select_mask != 0].mean(0).item(), select_mask.sum())
            # mask_prob.update(select_mask.mean().item())
            # total_c_teacher.update(total_acc.mean(0).item())

            p_hat = torch.zeros(batch_size, num_class).cuda().scatter_(1, p_hat.view(-1, 1), 1)
            select_mask = torch.cat([select_mask, select_mask], 0)

        with torch.no_grad():
            # todo: extract features from EMA and then xxx
            if args.ema_encoder:
                _, _, crt_feat_x = ema_model(crt_input_x, return_feature=True)
            else:
                _, _, crt_feat_x = model(crt_input_x, return_feature=True)
            crt_feat_x = crt_feat_x.squeeze()
            if args.ema_encoder:
                _, _, crt_feat_u = ema_model(crt_input_u, return_feature=True)
            else:
                _, _, crt_feat_u = model(crt_input_u, return_feature=True)
            crt_feat_u = crt_feat_u.squeeze()
            crt_feat_u_cls_probs = torch.softmax(crt_feat_u, dim=1)

            new_feat_list = []
            new_target_list = []
            for x, label_x, u in zip(crt_feat_x, crt_targets_x, crt_feat_u[:len(crt_targets_x)]):
                if random.random() < mixup_prob[label_x.argmax()]:
                    # lam = np.random.beta(args.crt_mix_alpha, args.crt_mix_alpha, size=1)
                    lam = np.random.uniform(args.max_lam, 1., size=1)
                    lam = torch.FloatTensor(lam).cuda()
                    # if args.maxlam:
                    #     lam = max(lam, 1 - lam)

                    if args.similar:
                        select_id = torch.multinomial(crt_feat_u_cls_probs[:, label_x.argmax()], 1)
                        u = crt_feat_u[select_id].view_as(x)

                    new_feat = lam * x + (1 - lam) * u
                    new_target = label_x
                else:
                    new_feat = x
                    new_target = label_x
                new_feat_list.append(new_feat)
                new_target_list.append(new_target)
            new_feat_tensor = torch.stack(new_feat_list, dim=0)  # [64, 128]
            new_target_tensor = torch.stack(new_target_list, dim=0)  # [64, 10]

        # todo: teacher forward
        teacher_logits = teacher_head(new_feat_tensor)
        teacher_loss = -torch.mean(torch.sum(F.log_softmax(teacher_logits, dim=1) * new_target_tensor, dim=1))
        # todo: log teacher_loss and teacher_acc
        teacher_optimizer.zero_grad()
        teacher_loss.backward()
        teacher_optimizer.step()
        ema_teacher_optimizer.step()

        with torch.no_grad():
            acc = (torch.argmax(teacher_logits, dim=1) == torch.argmax(crt_targets_x, dim=1)).float().mean()
            teacher_acc.update(acc.item(), crt_targets_x.size(0))
            teacher_acc.update(acc.item(), crt_targets_x.size(0))
            losses_teacher.update(teacher_loss.item(), crt_targets_x.size(0))

        all_inputs = torch.cat([inputs_x, inputs_u2, inputs_u3], dim=0)
        all_targets = torch.cat([targets_x, p_hat, p_hat], dim=0)

        all_outputs, _ = model(all_inputs)
        logits_x = all_outputs[:batch_size]
        logits_u = all_outputs[batch_size:]

        Lx, Lu = criterion(logits_x, all_targets[:batch_size], logits_u, all_targets[batch_size:], select_mask)
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

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | ' \
                     'Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f} | Loss_t: {loss_t:.4f} |' \
                     'Mask: {mask:.4f}| Use_acc: {used_acc:.4f} | teacher_acc: {teacher_acc:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    loss_t=losses_teacher.avg,
                    mask=mask_prob.avg,
                    used_acc=used_c.avg,
                    teacher_acc=teacher_acc.avg,
                    )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg, losses_teacher.avg, mask_prob.avg, total_c.avg, used_c.avg, teacher_acc.avg, emp_distb_u, pseudo_orig, pseudo_refine)

def validate_teacher(valloader, model, head, criterion, use_cuda, mode, log_name=None):

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
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            _, _, feats = model(inputs, return_feature=True)
            outputs = head(feats.squeeze())
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
    # prec = precision_score(y_true, y_pred, average=None)
    # recall = recall_score(y_true, y_pred, average=None)
    # print(prec)
    # print(recall)
    # with open(os.path.join(args.out, 'prec.json'), 'a') as f:
    #     json.dump(prec.tolist(), f)
    #     f.write(os.linesep)
    # with open(os.path.join(args.out, 'recall.json'), 'a') as f:
    #     json.dump(recall.tolist(), f)
    #     f.write(os.linesep)
    print(classwise_acc.tolist())
    print(classwise_acc.mean().tolist())
    if log_name is not None:
        with open(os.path.join(args.out, log_name), 'a') as f:
            json.dump(classwise_acc.tolist(), f)
            f.write(os.linesep)

    return (losses.avg, classwise_acc.mean().tolist(), section_acc.cpu().numpy(), GM)

def validate(valloader, model, criterion, use_cuda, mode, log_name=None):

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
    # prec = precision_score(y_true, y_pred, average=None)
    # recall = recall_score(y_true, y_pred, average=None)
    # print(prec)
    # print(recall)
    # with open(os.path.join(args.out, 'prec.json'), 'a') as f:
    #     json.dump(prec.tolist(), f)
    #     f.write(os.linesep)
    # with open(os.path.join(args.out, 'recall.json'), 'a') as f:
    #     json.dump(recall.tolist(), f)
    #     f.write(os.linesep)
    print(classwise_acc.tolist())
    print(classwise_acc.mean().tolist())
    if log_name is not None:
        with open(os.path.join(args.out, log_name), 'a') as f:
            json.dump(classwise_acc.tolist(), f)
            f.write(os.linesep)

    return (losses.avg, classwise_acc.mean().tolist(), section_acc.cpu().numpy(), GM)

def estimate_pseudo(q_y, saved_q):
    pseudo_labels = torch.zeros(len(saved_q), num_class)
    k_probs = torch.zeros(num_class)

    for i in range(1, num_class + 1):
        i = num_class - i
        num_i = int(args.alpha * q_y[i])
        sorted_probs, idx = saved_q[:, i].sort(dim=0, descending=True)
        pseudo_labels[idx[: num_i], i] = 1
        k_probs[i] = sorted_probs[:num_i].sum()

    return pseudo_labels, (q_y + 1e-6) / (k_probs + 1e-6)

def f(x, a, b, c, d):
    return np.sum(a * b * np.exp(-1 * x/c)) - d

def opt_solver(probs, target_distb, num_iter=args.iter_T, num_newton=30):
    entropy = (-1 * probs * torch.log(probs + 1e-6)).sum(1)
    weights = (1 / entropy)
    N, K = probs.size(0), probs.size(1)

    A, w, lam, nu, r, c = probs.numpy(), weights.numpy(), np.ones(N), np.ones(K), np.ones(N), target_distb.numpy()
    A_e = A / math.e
    X = np.exp(-1 * lam / w)
    Y = np.exp(-1 * nu.reshape(1, -1) / w.reshape(-1, 1))
    prev_Y = np.zeros(K)
    X_t, Y_t = X, Y

    for n in range(num_iter):
        # Normalization
        denom = np.sum(A_e * Y_t, 1)
        X_t = r / denom

        # Newton method
        Y_t = np.zeros(K)
        for i in range(K):
            Y_t[i] = optimize.newton(f, prev_Y[i], maxiter=num_newton, args=(A_e[:, i], X_t, w, c[i]), tol=1.0e-01)
        prev_Y = Y_t
        Y_t = np.exp(-1 * Y_t.reshape(1, -1) / w.reshape(-1, 1))

    denom = np.sum(A_e * Y_t, 1)
    X_t = r / denom
    M = torch.Tensor(A_e * X_t.reshape(-1, 1) * Y_t)

    return M

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

def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, mask):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1) * mask)

        return Lx, Lu

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        # self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            ema_param.data.copy_(param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param = ema_param.float()
            param = param.float()
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # # customized weight decay
            # param.mul_(1 - self.wd)

if __name__ == '__main__':
    main()