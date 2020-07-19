import os.path as osp

import cv2
import torch
from torch.utils.data import Dataset
import scipy.io
from .utils import Normalize
from .cutmix_utils import onehot, rand_bbox
import numpy as np
import random


class DogClfDataset(Dataset):

    def __init__(self, data_path, annotation_path, cutmix_beta, cutmix_prob, transforms=None, resizer=None,
                 means=(0.485, 0.456, 0.406), stds=(0.229, 0.224, 0.225), test_run=False):
        self.data_path = data_path
        self.annotation = scipy.io.loadmat(annotation_path)
        self.cutmix_beta = cutmix_beta
        self.cutmix_prob = cutmix_prob
        self.tramsforms = transforms
        self.resizer = resizer
        self.normalizer = Normalize(means, stds)

        if test_run:
            self.annotation['file_list'] = self.annotation['file_list'][:300]
            self.annotation['labels'] = self.annotation['labels'][:300]

    def __len__(self):
        return len(self.annotation['file_list'])

    def __getitem__(self, idx):
        path, label = self.annotation['file_list'][idx][0][0], self.annotation['labels'][idx][0]
        label = onehot(120, label - 1)
        image = cv2.imread(osp.join(self.data_path, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.tramsforms is not None:
            image = self.tramsforms(image=image)['image']

        if self.resizer:
            image, info = self.resizer(image)

        image = self.normalizer(image)

        if self.cutmix_beta >= 0 and self.cutmix_prob < np.random.rand(1):
            lam = np.random.beta(self.cutmix_beta, self.cutmix_beta)
            rand_index = random.choice(range(len(self)))

            path2, label2 = self.annotation['file_list'][rand_index][0][0], self.annotation['labels'][rand_index][0]
            label2 = onehot(120, label2 - 1)
            image2 = cv2.imread(osp.join(self.data_path, path2))
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

            if self.tramsforms is not None:
                image2 = self.tramsforms(image=image2)['image']

            if self.resizer:
                image2, info2 = self.resizer(image2)

            image2 = self.normalizer(image2)

            x1, y1, x2, y2 = rand_bbox(image2.size(), lam)
            image[:, x1:x2, y1:y2] = image2[:, x1:x2, y1:y2]
            lam = 1 - ((x2 - x1) * (y2 - y1) / (image.size()[-1] * image.size()[-2]))
            label = label * lam + label2 * (1. - lam)

        image = torch.FloatTensor(image)
        label = torch.FloatTensor(label)

        return {
            'image': image,
            'label': label
        }
