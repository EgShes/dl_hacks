import os
import os.path as osp
import uuid

import hydra
import torch
import torch.nn as nn
from albumentations import Compose, RandomBrightnessContrast, HorizontalFlip, ShiftScaleRotate, ToGray
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.data import DogClfDataset, Letterbox, VanillaResize
from src.models import resnet50
from src.training import RAdam, train_epoch, evaluate_epoch, fix_seeds, write2tensorboard, write2tensorboard_test


@hydra.main('train_config.yaml')
def train_model(args):

    fix_seeds(args.seed)

    experiment_name = f'{args.experiment_name}_{uuid.uuid4()}'
    if not args.test_run:
        writer = SummaryWriter(log_dir=osp.join('runs', experiment_name))
    device = torch.device(f'cuda:{args.gpu_num}')

    os.makedirs(osp.join(args.save_path, args.experiment_name), exist_ok=True)

    default_transforms = Compose([
        RandomBrightnessContrast(),
        HorizontalFlip(),
        ShiftScaleRotate(),
        ToGray(p=0.15)
    ])

    if args.resizer_type == 'vanilla':
        resizer = VanillaResize(args.height, args.width)
    elif args.resizer_type == 'letterbox':
        resizer = Letterbox(args.height, args.width)
    else:
        raise NotImplementedError(f'Resizer {args.resizer_type} not implemented')

    train_ds = DogClfDataset(
        osp.join(args.data_path, 'Images'), osp.join(args.data_path, 'lists', 'splitted_train_list.mat'),
        transforms=default_transforms, resizer=resizer, test_run=args.test_run
    )
    val_ds = DogClfDataset(
        osp.join(args.data_path, 'Images'), osp.join(args.data_path, 'lists', 'splitted_val_list.mat'), resizer=resizer,
        test_run=args.test_run
    )
    test_ds = DogClfDataset(osp.join(args.data_path, 'Images'), osp.join(args.data_path, 'lists', 'test_list.mat'),
                            resizer=resizer, test_run=args.test_run)

    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dl = DataLoader(val_ds, args.batch_size, num_workers=args.num_workers)
    test_dl = DataLoader(test_ds, args.batch_size, num_workers=args.num_workers)

    model = resnet50(pretrained=True, num_classes=120)
    model.to(device)

    optimizer = RAdam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    lr_scheduler = ReduceLROnPlateau(optimizer, factor=args.lr_factor, patience=args.lr_patience, verbose=True)

    best_loss, no_improvements = 1e10, 0
    for epoch in range(args.num_epochs):
        if no_improvements > args.early_stopping:
            break

        train_metrics = train_epoch(model, train_dl, optimizer, criterion, device)
        eval_metrics = evaluate_epoch(model, val_dl, criterion, device)

        if eval_metrics['loss'] < best_loss:
            best_loss = eval_metrics['loss']
            no_improvements = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'experiment_name': args.experiment_name,
                'writer_path': osp.join('runs', experiment_name)
            }, osp.join(args.save_path, f'{experiment_name}_best.pth'))
        else:
            no_improvements += 1

        lr_scheduler.step(eval_metrics['loss'])

        if not args.test_run:
            write2tensorboard(train_metrics, eval_metrics, writer, epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'])

    checkpoint = torch.load(osp.join(args.save_path, f'{experiment_name}_best.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate_epoch(model, test_dl, criterion, device)
    write2tensorboard_test(test_metrics, writer)


if __name__ == '__main__':
    train_model()
