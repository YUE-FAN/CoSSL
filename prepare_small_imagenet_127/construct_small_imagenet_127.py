from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.utils.data as data


small_imagenet_path = 'PATH TO SMALL Imagenet32/64'
small_imagenet_127_save_path = 'PATH TO SAVE YOUR SMALL Imagenet 127 32/64'


class SmallImageNet(data.Dataset):
    train_list = ['train_data_batch_1', 'train_data_batch_2', 'train_data_batch_3', 'train_data_batch_4',
                  'train_data_batch_5', 'train_data_batch_6', 'train_data_batch_7', 'train_data_batch_8',
                  'train_data_batch_9', 'train_data_batch_10']
    test_list = ['val_data']

    def __init__(self, file_path, imgsize, train, transform=None, target_transform=None):
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

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


# compute oldcls2newcls
oldcls2newcls = {}
path = './imgname2folder.txt'
with open(path, 'r') as f:
    txt = f.readlines()
for line in txt:
    img_name, folder = line.strip().split(' ')
    old_class = img_name[:-5].split('_')[0]
    if oldcls2newcls.get(old_class) is None:
        oldcls2newcls[old_class] = folder

# compute oldcls2idx
oldcls2idx = {}
idx2oldcls = {}
path = './map_clsloc.txt'
with open(path, 'r') as f:
    txt = f.readlines()
for line in txt:
    old_cls, idx, _ = line.split()
    oldcls2idx[old_cls] = int(idx) - 1
    idx2oldcls[int(idx) - 1] = old_cls
print(oldcls2idx)

# compute newcls2idx
newcls2idx = {}
path = './synset_words_up_down_127.txt'
with open(path, 'r') as f:
    txt = f.readlines()
for idx, line in enumerate(txt):
    new_cls, *_ = line.split()
    newcls2idx[new_cls] = idx
print(newcls2idx)

# generate ImageNet127_32/64
batch_list = ['train_data_batch_1', 'train_data_batch_2', 'train_data_batch_3', 'train_data_batch_4',
              'train_data_batch_5', 'train_data_batch_6', 'train_data_batch_7', 'train_data_batch_8',
              'train_data_batch_9', 'train_data_batch_10', 'val_data']

for filename in batch_list:
    file = os.path.join(small_imagenet_path, filename)
    with open(file, 'rb') as f:
        if sys.version_info[0] == 2:
            entry = pickle.load(f)
        else:
            entry = pickle.load(f, encoding='latin1')

    new_targets = []
    for ll in entry['labels']:
        old_cls = idx2oldcls[ll - 1]  # Labels are indexed from 1
        new_cls = oldcls2newcls[old_cls]
        tar = newcls2idx[new_cls]
        new_targets.append(tar + 1)
    print(set(new_targets))
    if filename == 'val_data':
        d = {'data': entry['data'], 'labels': new_targets}
    else:
        d = {'data': entry['data'], 'mean': entry['mean'], 'labels': new_targets}
    pickle.dump(d, open(os.path.join(small_imagenet_127_save_path, filename), 'wb'))
