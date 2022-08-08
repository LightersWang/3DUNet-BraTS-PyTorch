import math

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, MultiStepLR

def get_scheduler(args, optimizer: torch.optim):
    """Generate the learning rate scheduler for **every epoch**

    Args:
        optimizer (torch.optim): Optimizer
        epochs (int): training epochs

    Returns:
        lr_scheduler
    """
    epochs = args.epochs

    if args.scheduler == 'warmup_cosine':
        warmup = args.warmup_epochs
        warmup_cosine_lr = (lambda epoch: epoch / warmup 
            if epoch <= warmup else 0.5 * (
                math.cos((epoch - warmup) / (epochs - warmup) * math.pi) + 1))
        lr_scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine_lr)
    elif args.scheduler == 'cosine':
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    elif args.scheduler == 'step':
        lr_scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_gamma)
    elif args.scheduler == 'poly':
        lr_scheduler = LambdaLR(optimizer, lambda epoch: (1 - epoch / epochs) ** 0.9)
    elif args.scheduler == 'none':
        lr_scheduler = None
    else:
        raise NotImplementedError(f"LR scheduler {args.scheduler} is not implemented.")

    return lr_scheduler