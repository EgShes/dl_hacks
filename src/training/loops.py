from collections import defaultdict

import torch
from numpy.random import rand, uniform
from tqdm import tqdm

from src.data import perform_fgsm_attack

from .metrics import accuracy, macro_average_precision


def train_epoch(model, loader, optimizer, criterion, device, fgsm_prob, fgsm_eps):
    model.train()

    epoch_metrics = defaultdict(lambda: 0.)
    for batch in tqdm(loader, desc='Training', total=len(loader)):
        if rand() <= fgsm_prob:
            batch = perform_fgsm_attack(
                model, batch, criterion, uniform(*fgsm_eps), device
            )
        else:
            batch = {key: val.to(device) for key, val in batch.items()}

        optimizer.zero_grad()

        pred = model(batch['image'])
        loss = criterion(pred, batch['label'])
        loss.backward()
        optimizer.step()

        epoch_metrics['loss'] += loss.item()
        epoch_metrics['accuracy'] += accuracy(pred, batch['label'])
        epoch_metrics['macro_map'] += macro_average_precision(pred, batch['label'])

    return {key: val / len(loader) for key, val in epoch_metrics.items()}


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()

    epoch_metrics = defaultdict(lambda: 0.)
    for batch in tqdm(loader, desc='Evaluation', total=len(loader)):
        batch = {key: val.to(device) for key, val in batch.items()}

        pred = model(batch['image'])
        loss = criterion(pred, batch['label'])

        epoch_metrics['loss'] += loss.item()
        epoch_metrics['accuracy'] += accuracy(pred, batch['label'])
        epoch_metrics['macro_map'] += macro_average_precision(pred, batch['label'])

    return {key: val / len(loader) for key, val in epoch_metrics.items()}
