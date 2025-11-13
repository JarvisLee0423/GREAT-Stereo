import logging
import argparse
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class Logger:

    SUM_FREQ = 100

    def __init__(self, args: argparse.Namespace, model: nn.Module, scheduler: optim.lr_scheduler._LRScheduler):
        self.args = args
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = SummaryWriter(log_dir=self.args.logdir)
    
    def print_training_status(self) -> None:
        metrics_data = []
        if "depthany" in self.args.name or "monster" in self.args.name:
            training_str = "[{:6d}, {:10.7f}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0], self.scheduler.get_last_lr()[1])
        else:
            training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        for key in sorted(self.running_loss.keys()):
            line = key + f": {round(self.running_loss[key] / Logger.SUM_FREQ, 4)}"
            metrics_data.append(line)
        metrics_str = " | ".join(metrics_data)

        # Print the training status.
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")
        with open(self.args.logdir + "/training_settings.txt", "a") as file:
            file.write(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}\n")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.args.logdir)
        
        for key in self.running_loss:
            self.writer.add_scalar(key, self.running_loss[key] / Logger.SUM_FREQ, self.total_steps)
            self.running_loss[key] = 0.0
    
    def push(self, metrics: dict) -> None:
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0
            
            self.running_loss[key] += metrics[key]
        
        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
            self.print_training_status()
            self.running_loss = {}
    
    def write_dict(self, results: dict) -> None:
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.args.logdir)
        
        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)
    
    def close(self) -> None:
        self.writer.close()
