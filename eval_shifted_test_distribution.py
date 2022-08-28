"""
This script tests models trained on CIFAR under
skewed class distributions at test time. It follows the
idea from "Disentangling Label Distribution for Long-tailed
Visual Recognition".
"""
from __future__ import print_function

import argparse
import os
import time
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision
from torchvision.transforms import transforms

import models.wrn as models
from utils import Bar, AverageMeter, accuracy

parser = argparse.ArgumentParser(description='PyTorch Eval')
parser.add_argument('--dataset', type=str, help='cifar10 or cifar100')
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--gpu', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES')

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

if args.dataset == 'cifar10':
    num_classes = 10
elif args.dataset == 'cifar100':
    num_classes = 100
else:
    raise NotImplementedError

imbalance_ratio_list = [512, 256, 150, 128, 64, 32, 16, 8, 4, 2, 1, -2, -4, -8, -16, -32, -64, -128, -256, -512]
model_path_list = [

# add paths to your checkpoints
    
]


def make_imb_data(max_num, class_num, gamma):
    mu = np.power(1/abs(gamma), 1/(class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / abs(gamma)))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    if gamma < 0:
        class_num_list = class_num_list[::-1]
    return list(class_num_list)


def train_split(labels, n_labeled_per_class, num_classes):
    labels = np.array(labels)
    test_idxs = []
    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        test_idxs.extend(idxs[:n_labeled_per_class[i]])  # only take the first n samples
    return test_idxs


class CIFAR100_customize(torchvision.datasets.CIFAR100):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR100_customize, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = [Image.fromarray(img) for img in self.data]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class CIFAR10_customize(torchvision.datasets.CIFAR10):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_customize, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
        self.data = [Image.fromarray(img) for img in self.data]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


def create_model():
    model = models.WRN(2, num_classes).cuda()
    teacher = nn.Linear(model.output.in_features, num_classes, bias=True).cuda()
    return model, teacher


def main():
    test_acc_list = [[model_path] for model_path in model_path_list]
    for imbalance_ratio in imbalance_ratio_list:
        for ii, model_path in enumerate(model_path_list):

            if args.dataset == 'cifar10':
                num_max = 1000
                data_mean = (0.4914, 0.4822, 0.4465)
                data_std = (0.2471, 0.2435, 0.2616)
                data_path = '/BS/databases00/cifar-10'
                base_test_set_constructor = torchvision.datasets.CIFAR10
                test_set_constructor = CIFAR10_customize
            elif args.dataset == 'cifar100':
                num_max = 100
                data_mean = (0.5071, 0.4867, 0.4408)
                data_std = (0.2675, 0.2565, 0.2761)
                data_path = '/BS/databases00/cifar-100'
                base_test_set_constructor = torchvision.datasets.CIFAR100
                test_set_constructor = CIFAR10_customize
            else:
                raise NotImplementedError

            transform_val = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(data_mean, data_std)
            ])
            base_test_set = base_test_set_constructor(data_path, train=False, transform=transform_val, download=False)
            N_SAMPLES_PER_CLASS = make_imb_data(num_max, num_classes, imbalance_ratio)
            test_idxs = train_split(base_test_set.targets, N_SAMPLES_PER_CLASS, num_classes)
            test_set = test_set_constructor(data_path, test_idxs, train=False, transform=transform_val)
            test_loader = data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=4)

            model, teacher = create_model()

            assert os.path.isfile(os.path.join(model_path, 'checkpoint.pth.tar')), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(os.path.join(model_path, 'checkpoint.pth.tar'))
            model.load_state_dict(checkpoint['ema_state_dict'])

            if 'cossl' in model_path:
                teacher.load_state_dict(checkpoint['ema_teacher'])
                _, test_acc, _, _ = validate_cossl(test_loader, model, teacher, nn.CrossEntropyLoss(), use_cuda, num_classes, 'eval')
            else:
                _, test_acc, _, _ = validate(test_loader, model, nn.CrossEntropyLoss(), use_cuda, num_classes, 'eval')

            print(imbalance_ratio, test_acc, model_path)
            test_acc_list[ii].append(test_acc)
    print(test_acc_list)


def validate(valloader, model, criterion, use_cuda, num_classes, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    classwise_correct = torch.zeros(num_classes).cuda()
    classwise_num = torch.zeros(num_classes).cuda()
    section_acc = torch.zeros(3).cuda()

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

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
            pred_mask = (targets == pred_label).float()
            for i in range(num_classes):
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
                        top5=top5.avg)
            bar.next()
        bar.finish()

    # Major, Neutral, Minor
    section_num = int(num_classes / 3)
    classwise_acc = (classwise_correct / classwise_num)
    section_acc[0] = classwise_acc[:section_num].mean()
    section_acc[2] = classwise_acc[-1 * section_num:].mean()
    section_acc[1] = classwise_acc[section_num:-1 * section_num].mean()
    GM = 1
    for i in range(num_classes):
        if classwise_acc[i] == 0:
            # To prevent the N/A values, we set the minimum value as 0.001
            GM *= (1/(100 * num_classes)) ** (1/num_classes)
        else:
            GM *= (classwise_acc[i]) ** (1/num_classes)

    return (losses.avg, top1.avg, section_acc.cpu().numpy(), GM)


def validate_cossl(valloader, model, head, criterion, use_cuda, num_classes, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))

    classwise_correct = torch.zeros(num_classes).cuda()
    classwise_num = torch.zeros(num_classes).cuda()
    section_acc = torch.zeros(3).cuda()

    with torch.no_grad():
        for batch_idx, (inputs, targets, _) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

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
            pred_mask = (targets == pred_label).float()
            for i in range(num_classes):
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
    section_num = int(num_classes / 3)
    classwise_acc = (classwise_correct / classwise_num)
    section_acc[0] = classwise_acc[:section_num].mean()
    section_acc[2] = classwise_acc[-1 * section_num:].mean()
    section_acc[1] = classwise_acc[section_num:-1 * section_num].mean()
    GM = 1
    for i in range(num_classes):
        if classwise_acc[i] == 0:
            # To prevent the N/A values, we set the minimum value as 0.001
            GM *= (1/(100 * num_classes)) ** (1/num_classes)
        else:
            GM *= (classwise_acc[i]) ** (1/num_classes)

    return (losses.avg, top1.avg, section_acc.cpu().numpy(), GM)


if __name__ == '__main__':
    main()