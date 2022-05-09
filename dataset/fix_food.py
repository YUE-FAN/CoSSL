import torch.utils.data as data
import numpy as np
import os
import sys
import json
from torchvision.datasets import folder as dataset_parser
from torchvision.transforms import transforms

from dataset.randaugment import RandAugmentMC


class TransformTwice:
    def __init__(self, transform, transform2):
        self.transform = transform
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform2(inp)
        return out1, out2


def get_food101(root, l_samples, u_samples, seed=0, return_strong_labeled_set=False):

    imgnet_mean = (0.485, 0.456, 0.406)
    imgnet_std = (0.229, 0.224, 0.225)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_strong = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        RandAugmentMC(n=2, m=10, img_size=224),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(imgnet_mean, imgnet_std)
    ])

    base_dataset = Food101(root, split='train')
    train_labeled_idxs, train_unlabeled_idxs = train_split(base_dataset.targets, l_samples, u_samples, 101, seed)

    train_labeled_dataset = Food101(root, split='train', indexs=train_labeled_idxs, transform=transform_train)
    train_unlabeled_dataset = Food101(root, split='train', indexs=train_unlabeled_idxs,
                                      transform=TransformTwice(transform_train, transform_strong))
    test_dataset = Food101(root, split='test', transform=transform_val)

    print(f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)}")
    if return_strong_labeled_set:
        train_strong_labeled_dataset = Food101(root, split='train', indexs=train_labeled_idxs, transform=transform_strong)
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_strong_labeled_dataset
    else:
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def train_split(labels, n_labeled_per_class, n_unlabeled_per_class, num_classes, seed):
    np.random.seed(seed)
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []

    for i in range(num_classes):
        idxs = np.where(labels == i)[0]
        if seed != 0:
            np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class[i]])
        # train_unlabeled_idxs.extend(idxs[:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class[i]:n_labeled_per_class[i] + n_unlabeled_per_class[i]])
        # note here train_labeled_idxs and train_unlabeled_idxs are disjoint!!!

    return train_labeled_idxs, train_labeled_idxs + train_unlabeled_idxs


def make_dataset(dir, class_to_idx, json_file, ):
    with open(json_file, 'r') as f:
        tmp = json.load(f)

    images = []
    for class_name, file_list in tmp.items():
        target = class_to_idx[class_name]
        for file_name in file_list:
            img_path = os.path.join(dir, file_name + '.jpg')
            images.append((img_path, target))
    return images


class Food101(data.Dataset):
    def __init__(self, dataset_root, split, transform=None, target_transform=None,
                 loader=dataset_parser.default_loader, indexs=None):
        assert split in ['train', 'test']

        self.dataset_root = dataset_root
        self.image_root = os.path.join(dataset_root, 'images')
        split2json = {
            'train': os.path.join(dataset_root, 'meta/train.json'),
            'test': os.path.join(dataset_root, 'meta/test.json')
        }

        self.classes, self.class_to_idx = self._find_classes(self.image_root)
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

        self.data = make_dataset(self.image_root, self.class_to_idx, split2json[split])
        self.targets = [s[1] for s in self.data]

        if indexs is not None:
            samples = [self.data[i] for i in indexs]
            self.data = samples
            self.targets = np.array(self.targets)[indexs]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.data[index]
        img_original = self.loader(path)

        if self.transform is not None:
            img = self.transform(img_original)
        else:
            img = img_original.copy()

        return img, target, index

    def __len__(self):
        return len(self.data)

