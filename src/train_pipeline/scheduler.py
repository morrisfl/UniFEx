import torch
import torch.optim as optim
import pytorch_warmup as warmup

from train_pipeline.optimizer import SAM


def build_scheduler(config, optimizer):
    if isinstance(optimizer, SAM):
        optimizer = optimizer.base_optimizer

    if config.SCHEDULER.name is None:
        return None
    elif config.SCHEDULER.name == "cosine":
        if config.SCHEDULER.epoch_based:
            steps = config.TRAIN.epochs
        else:
            steps = config.TRAIN.iterations
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=config.SCHEDULER.min_lr)
    else:
        raise NotImplementedError(f"Scheduler {config.SCHEDULER.name} not implemented.")


def build_warmup_scheduler(config, optimizer):
    if isinstance(optimizer, SAM):
        optimizer = optimizer.base_optimizer

    if config.SCHEDULER.warmup is None:
        return None
    elif config.SCHEDULER.warmup == "linear":
        return warmup.LinearWarmup(optimizer, warmup_period=config.SCHEDULER.warmup_steps)
    elif config.SCHEDULER.warmup == "exponential":
        return warmup.ExponentialWarmup(optimizer, warmup_period=config.SCHEDULER.warmup_steps)
    else:
        raise NotImplementedError(f"Warmup scheduler {config.SCHEDULER.warmup} not implemented.")
