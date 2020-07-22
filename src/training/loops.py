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

        cls_logits, confidence_logits = model(batch['image'])

        cls_probs = torch.softmax(cls_logits, -1)
        confidence_probs = torch.sigmoid(confidence_logits)

        cls_probs, confidence_probs = model.clamp([cls_probs, confidence_probs], 1e-12)

        cls_probs, confidence_loss = model.use_confidence(cls_probs, confidence_probs, batch['label'])

        loss = criterion(cls_probs, batch['label']) + model.confidence_lambda * confidence_loss

        model.confidence_lambda = model.calc_new_lambda(model.confidence_lambda, model.confidence_budget, confidence_loss)

        loss.backward()
        optimizer.step()

        epoch_metrics['loss'] += loss.item()
        epoch_metrics['accuracy'] += accuracy(cls_logits, batch['label'])
        epoch_metrics['macro_map'] += macro_average_precision(cls_logits, batch['label'])

    return {key: val / len(loader) for key, val in epoch_metrics.items()}


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()

    epoch_metrics = defaultdict(lambda: 0.)
    for batch in tqdm(loader, desc='Evaluation', total=len(loader)):
        batch = {key: val.to(device) for key, val in batch.items()}

        cls_logits, confidence_logits = model(batch['image'])
        loss = criterion(cls_logits, batch['label'])

        epoch_metrics['loss'] += loss.item()
        epoch_metrics['accuracy'] += accuracy(cls_logits, batch['label'])
        epoch_metrics['macro_map'] += macro_average_precision(cls_logits, batch['label'])

    return {key: val / len(loader) for key, val in epoch_metrics.items()}
