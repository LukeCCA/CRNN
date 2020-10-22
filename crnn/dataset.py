#!/usr/bin/python
# encoding: utf-8

import random
import cv2
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import sys
import numpy as np


class lmdbDataset(Dataset):

    def __init__(self, root=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()).decode())
            self.nSamples = nSamples

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            input_key = 'input-%05d' % index
            inputbuf = txn.get(input_key.encode())
            input_data = np.frombuffer(inputbuf, dtype=np.float64)

            label_key = 'label-%05d' % index
            label = txn.get(label_key.encode()).decode()

        return (input_data, label)

# resize each image


class resizeNormalize:

    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, input_data):
        img = cv2.resize(
            input_data,
            self.size,
            interpolation=self.interpolation)
        img = self.toTensor(img)
        return img

######### 可以改寫的 #########


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = self.num_samples // self.batch_size
        tail = self.num_samples % self.batch_size
        index = torch.LongTensor(self.num_samples).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(
                0, self.num_samples - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(
                0, self.num_samples - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples
###########################


class alignCollate:

    def __init__(self, H=50, W=200):
        self.H = H
        self.W = W

    def __call__(self, batch):
        inputs, labels = zip(*batch)
        H = self.H
        W = self.W
        transform = resizeNormalize((W, H))
        inputs = [transform(input_data) for input_data in inputs]
        inputs = torch.cat([t.unsqueeze(0) for t in inputs], 0)

        return inputs, labels
