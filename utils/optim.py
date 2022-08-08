import torch
import torch.nn as nn

def get_optimizer(args, net: nn.Module):
    lr, weight_decay = args.lr, args.weight_decay

    if args.optim == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), lr, weight_decay=weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr, momentum=args.beta1, nesterov=True)
    else:
        optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=weight_decay)     # default is adam

    return optimizer
