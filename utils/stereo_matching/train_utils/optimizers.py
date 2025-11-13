import argparse
import torch.nn as nn
import torch.optim as optim
from typing import Tuple


def fetch_adamw_optimizer(args: argparse.Namespace, model: nn.Module) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    """ Create the optimizer and learning rate scheduler. """
    if "depthany" in args.name or "monster" in args.name:
        print("Separating the parameters of the feature encoder and rest of the model...")
        feat_decoder_params = list(map(id, model.feat_decoder.parameters()))
        rest_params = filter(lambda x: id(x) not in feat_decoder_params and x.requires_grad, model.parameters())
        params_dict = [
            {"params": model.feat_decoder.parameters(), "lr": args.lr / 2.0},
            {"params": rest_params, "lr": args.lr},
        ]
        print("Done.")

        optimizer = optim.AdamW(params_dict, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            [args.lr / 2.0, args.lr],
            args.num_steps + 100,
            pct_start=0.01,
            cycle_momentum=False,
            anneal_strategy="linear",
        )
    else:
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
