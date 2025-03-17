import argparse
import torch.nn as nn
import torch.optim as optim
from typing import Tuple


def fetch_adamw_optimizer(args: argparse.Namespace, model: nn.Module) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """ Create the optimizer and learning rate scheduler. """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        args.lr,
        args.num_steps + 100,
        pct_start=0.01,
        cycle_momentum=False,
        anneal_strategy="linear",
    )

    return optimizer, scheduler
