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
        loss = torch.tensor(0.).float().to(device)
        for _, value in pred.items():
            loss = loss + criterion(value, batch['label'])
        loss = torch.div(loss, len(pred))
        loss.backward()
        optimizer.step()

        epoch_metrics['loss'] += loss.item()
        epoch_metrics['accuracy'] += accuracy(pred['out4'], batch['label'])
        epoch_metrics['macro_map'] += macro_average_precision(pred['out4'], batch['label'])

    return {key: val / len(loader) for key, val in epoch_metrics.items()}


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()

    epoch_metrics = defaultdict(lambda: 0.)
    for batch in tqdm(loader, desc='Evaluation', total=len(loader)):
        batch = {key: val.to(device) for key, val in batch.items()}

        pred = model(batch['image'])
        loss = torch.tensor(0.).float().to(device)
        for _, value in pred.items():
            loss += criterion(value, batch['label'])
        loss = torch.div(loss, len(pred))

        epoch_metrics['loss'] += loss.item()
        epoch_metrics['accuracy'] += accuracy(pred['out4'], batch['label'])
        epoch_metrics['macro_map'] += macro_average_precision(pred['out4'], batch['label'])

    return {key: val / len(loader) for key, val in epoch_metrics.items()}
