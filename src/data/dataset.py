import os.path as osp

import cv2
import torch
from torch.utils.data import Dataset
import scipy.io
from .utils import Normalize


class DogClfDataset(Dataset):

    def __init__(self, data_path, annotation_path, transforms=None, resizer=None, means=(0.485, 0.456, 0.406),
                 stds=(0.229, 0.224, 0.225), test_run=False):
        self.data_path = data_path
        self.annotation = scipy.io.loadmat(annotation_path)
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
        image = cv2.imread(osp.join(self.data_path, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.tramsforms is not None:
            image = self.tramsforms(image=image)['image']

        if self.resizer:
            image, info = self.resizer(image)

        image = self.normalizer(image)

        image = torch.FloatTensor(image)
        label = torch.tensor(label - 1).long()

        return {
            'image': image,
            'label': label
        }
