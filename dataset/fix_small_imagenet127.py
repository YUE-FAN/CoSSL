import os
import sys
import pickle
import numpy as np
from PIL import Image
import torch.utils.data as data

import torchvision
import torch
from torchvision.transforms import transforms
from RandAugment import RandAugment
from RandAugment.augmentations import CutoutDefault


class TransformTwice:
    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        out3 = self.transform2(inp)
        return out1, out2, out3


def get_small_imagenet(root, img_size, labeled_percent=0.1, seed=0, return_strong_labeled_set=False):

    assert img_size == 32 or img_size == 64, 'img size should only be 32 or 64!!!'
    base_dataset = SmallImageNet(root, img_size, True)

    # compute dataset mean and std
    dataset_mean = (0.48109809, 0.45747185, 0.40785507)  # np.mean(base_dataset.data, axis=(0, 1, 2)) / 255
    print(dataset_mean)

    dataset_std = (0.26040889, 0.2532126, 0.26820634)  # np.std(base_dataset.data, axis=(0, 1, 2)) / 255
    print(dataset_std)

    # construct data augmentation
    # Augmentations.
    transform_train = transforms.Compose([
        transforms.RandomCrop(img_size, padding=int(img_size / 8)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

    transform_strong = transforms.Compose([
        transforms.RandomCrop(img_size, padding=int(img_size / 8)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])
    transform_strong.transforms.insert(0, RandAugment(3, 4))
    transform_strong.transforms.append(CutoutDefault(int(img_size / 2)))

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(dataset_mean, dataset_std)
    ])

    # select labeled data and construct labeled dataset
    num_classes = len(set(base_dataset.targets))
    num_data_per_cls = [0 for _ in range(num_classes)]
    for l in base_dataset.targets:
        num_data_per_cls[l] += 1

    num_labeled_data_per_cls = [int(np.around(n * labeled_percent)) for n in num_data_per_cls]
    print('total number of labeled data is ', sum(num_labeled_data_per_cls))
    train_labeled_idxs = train_split(base_dataset.targets, num_labeled_data_per_cls, num_classes, seed)

    train_labeled_dataset = SmallImageNet(root, img_size, True, transform=transform_train, indexs=train_labeled_idxs)
    train_unlabeled_dataset = SmallImageNet(root, img_size, True, transform=TransformTwice(transform_train, transform_strong))
    test_dataset = SmallImageNet(root, img_size, False, transform=transform_val)

    if return_strong_labeled_set:
        train_strong_labeled_dataset = SmallImageNet(root, img_size, True, transform=transform_strong, indexs=train_labeled_idxs)
        return num_data_per_cls, train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_strong_labeled_dataset
    else:
        return num_data_per_cls, train_labeled_dataset, train_unlabeled_dataset, test_dataset


def train_split(labels, n_labeled_per_class, num_classes, seed):
    np.random.seed(seed)
    labels = np.array(labels)
    train_labeled_idxs = []
    # train_unlabeled_idxs = []

    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        if seed != 0:
            np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        # train_unlabeled_idxs.extend(idxs[:n_labeled_per_class[i] + n_unlabeled_per_class[i]])

    return train_labeled_idxs  # , train_unlabeled_idxs


class SmallImageNet(data.Dataset):
    train_list = ['train_data_batch_1', 'train_data_batch_2', 'train_data_batch_3', 'train_data_batch_4',
                  'train_data_batch_5', 'train_data_batch_6', 'train_data_batch_7', 'train_data_batch_8',
                  'train_data_batch_9', 'train_data_batch_10']
    test_list = ['val_data']

    def __init__(self, file_path, imgsize, train, transform=None, target_transform=None, indexs=None):
        # assert imgsize == 32 or imgsize == 64, 'imgsize should only be 32 or 64'
        self.imgsize = imgsize
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.targets = []
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        # now load the picked numpy arrays
        for filename in downloaded_list:
            file = os.path.join(file_path, filename)
            with open(file, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])  # Labels are indexed from 1
        self.targets = [i - 1 for i in self.targets]
        self.data = np.vstack(self.data).reshape((len(self.targets), 3, self.imgsize, self.imgsize))
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)
