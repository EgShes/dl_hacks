from collections import defaultdict

import torch
from tqdm import tqdm

from .metrics import accuracy, macro_average_precision


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()

    epoch_metrics = defaultdict(lambda: 0.)
    for batch in tqdm(loader, desc='Training', total=len(loader)):
        batch = {key: val.to(device) for key, val in batch.items()}

        optimizer.zero_grad()

        pred = model(batch['image'])
        loss = criterion(pred, batch['label'])
        loss.backward()
        optimizer.step()

        epoch_metrics['loss'] += loss.item()
        epoch_metrics['accuracy'] += accuracy(pred, torch.argmax(batch['label'], -1))
        epoch_metrics['macro_map'] += macro_average_precision(pred, torch.argmax(batch['label'], -1))

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
        epoch_metrics['accuracy'] += accuracy(pred, torch.argmax(batch['label'], -1))
        epoch_metrics['macro_map'] += macro_average_precision(pred, torch.argmax(batch['label'], -1))

    return {key: val / len(loader) for key, val in epoch_metrics.items()}
