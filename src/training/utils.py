import torch
import random
import numpy as np


def fix_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def write2tensorboard(train_metrics, val_metrics, writer, epoch):
    for key in train_metrics:
        train_val, val_val = train_metrics[key], val_metrics[key]
        writer.add_scalar(f'{key}/{key}_train', train_val, epoch)
        writer.add_scalar(f'{key}/{key}_val', val_val, epoch)
