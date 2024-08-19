import math
import random


def generate_name(config):
    model_id = random.randint(1, 99999)
    backbone = f"{config.MODEL.BACKBONE.type}-{config.MODEL.BACKBONE.network_arch.lower()}"
    head = f"{config.MODEL.HEAD.name.lower()}{config.MODEL.HEAD.k}"
    transforms = f"{config.TRANSFORM.name}-aug"
    optimizer = config.OPTIMIZER.name.lower()
    learning_rate = config.OPTIMIZER.lr
    scheduler = config.SCHEDULER.name
    warmup = config.SCHEDULER.warmup
    if scheduler is None:
        name = f"m{model_id:05d}_{backbone}_{head}_{transforms}_{optimizer}{learning_rate:.0e}"
    else:
        if warmup is None:
            name = f"m{model_id:05d}_{backbone}_{head}_{transforms}_{optimizer}{learning_rate:.0e}_{scheduler}"
        else:
            name = f"m{model_id:05d}_{backbone}_{head}_{transforms}_{optimizer}{learning_rate:.0e}_{scheduler}_{warmup}"
    return name


def calc_training_params(config, num_samples):
    batches_per_epoch = math.ceil(num_samples / config.DATALOADER.batch_size)
    if config.TRAIN.epoch_based:
        config.TRAIN.iterations = batches_per_epoch * config.TRAIN.epochs

    if config.SCHEDULER.warmup is not None:
        if config.SCHEDULER.warmup_steps < 10:
            config.SCHEDULER.warmup_steps = batches_per_epoch * config.SCHEDULER.warmup_steps
        else:
            config.SCHEDULER.warmup_steps = int(config.seen_img_warmup / config.DATALOADER.batch_size)

    config.TRAIN.save_iter = int(config.TRAIN.seen_img_save / config.DATALOADER.batch_size)


if __name__ == "__main__":
    pass
