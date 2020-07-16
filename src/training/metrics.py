import torch
from sklearn.metrics import precision_score


@torch.no_grad()
def accuracy(pred, gt):
    acc = (torch.argmax(pred, -1) == gt).sum().item() / len(gt)
    return acc

@torch.no_grad()
def macro_average_precision(pred, gt):
    pred = torch.argmax(pred, -1)
    map = precision_score(gt.cpu().numpy(), pred.cpu().numpy(), average='macro')
    return map
