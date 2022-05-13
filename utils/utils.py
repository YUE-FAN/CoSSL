from __future__ import print_function

import os
import copy
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import WeightedRandomSampler


__all__ = ['make_imb_data', 'save_checkpoint', 'MixMatch_Loss', 'FixMatch_Loss',
           'ReMixMatch_Loss', 'linear_rampup', 'get_weighted_sampler', 'merge_two_datasets',
           'WeightEMA', 'interleave']


def make_imb_data(max_num, class_num, gamma):
    mu = np.power(1 / abs(gamma), 1 / (class_num - 1))
    class_num_list = []
    for i in range(class_num):
        if i == (class_num - 1):
            class_num_list.append(int(max_num / abs(gamma)))
        else:
            class_num_list.append(int(max_num * np.power(mu, i)))
    if gamma < 0:
        class_num_list = class_num_list[::-1]
    # print(class_num_list)
    return list(class_num_list)


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class ReMixMatch_Loss(object):
    def __init__(self, lambda_u, rampup_length):
        self.lambda_u = lambda_u
        self.rampup_length = rampup_length

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1))

        return Lx, Lu, self.lambda_u * linear_rampup(epoch, self.rampup_length)


class MixMatch_Loss(object):
    def __init__(self, lambda_u, rampup_length):
        self.lambda_u = lambda_u
        self.rampup_length = rampup_length

    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, self.lambda_u * linear_rampup(epoch, self.rampup_length)


class FixMatch_Loss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, mask):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * targets_u, dim=1) * mask)
        return Lx, Lu


class WeightEMA(object):
    def __init__(self, model, ema_model, lr, alpha=0.999, wd=True):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        if wd:
            self.wd = 0.02 * lr
        else:
            self.wd = 0.0

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param = ema_param.float()
            param = param.float()
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


def save_checkpoint(state, epoch, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

    if epoch % 100 == 0:
        shutil.copyfile(filepath, os.path.join(checkpoint, f'checkpoint_{epoch + 1}.pth.tar'))


def get_weighted_sampler(target_sample_rate, num_sample_per_class, target):
    assert len(num_sample_per_class) == len(np.unique(target))

    sample_weights = target_sample_rate / num_sample_per_class  # this is the key line!!!

    # assign each sample a weight by sampling rate
    samples_weight = np.array([sample_weights[t] for t in target])

    return WeightedRandomSampler(samples_weight, len(samples_weight), True)


class merge_two_datasets(torch.utils.data.Dataset):
    def __init__(self, data1, data2, targets1, targets2,
                 transform=None, target_transform=None):
        self.data = copy.deepcopy(data1 + data2)
        self.targets = copy.deepcopy(np.concatenate([targets1, targets2], axis=0))
        assert len(self.data) == len(self.targets)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)