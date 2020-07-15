import hydra
import scipy.io
from numpy.random import permutation
import numpy as np
import os.path as osp
from copy import deepcopy


@hydra.main('../../train_config.yaml')
def split(args):
    np.random.seed(args.seed)

    orig_train_data = scipy.io.loadmat(osp.join(args.data_path, 'lists', 'train_list.mat'))
    orig_train_length = len(orig_train_data['file_list'])

    indicies = list(range(100))
    indicies = permutation(indicies)
    train_indices = set(indicies[:85])
    val_indices = set(indicies[85:])

    train_indices = np.array([i for i in range(orig_train_length) if i % 100 in train_indices])
    val_indices = np.array([i for i in range(orig_train_length) if i % 100 in val_indices])

    train_data = deepcopy(orig_train_data)
    val_data = deepcopy(orig_train_data)

    train_data['file_list'] = train_data['file_list'][train_indices]
    train_data['annotation_list'] = train_data['annotation_list'][train_indices]
    train_data['labels'] = train_data['labels'][train_indices]

    val_data['file_list'] = val_data['file_list'][val_indices]
    val_data['annotation_list'] = val_data['annotation_list'][val_indices]
    val_data['labels'] = val_data['labels'][val_indices]

    scipy.io.savemat(osp.join(args.data_path, 'lists', 'splitted_train_list.mat'), train_data)
    scipy.io.savemat(osp.join(args.data_path, 'lists', 'splitted_val_list.mat'), val_data)


if __name__ == '__main__':
    split()