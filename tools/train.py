import argparse
import datetime
import os
import time
import shutil

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import build_dataset, get_num_samples_per_cls
from model import TrainModel
from train_pipeline import iter_based_training, epoch_based_training, set_random_seed
from train_pipeline import build_optimizer, build_scheduler, build_warmup_scheduler, build_loss, TrainTracker
from utils import get_cfg_defaults, setup_logger, dump_yaml_to_log, generate_name, calc_training_params


def pars_args():
    parser = argparse.ArgumentParser(description="Train multi-domain image embedding model.")
    parser.add_argument("config_file", help="Path to config file")
    parser.add_argument("data_root", help="Path to the root of the datasets")
    parser.add_argument("--output_dir", default="results/", help="Path to output dir")
    parser.add_argument("--data_parallelism", action="store_true", help="Whether to use multiple GPU's")
    parser.add_argument("--device", default="cuda:0", nargs="?", help="Specify the GPU")

    return parser.parse_args()


if __name__ == '__main__':
    args = pars_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)

    cfg.DATASET.root = args.data_root
    cfg.TRAIN.device = args.device

    name = generate_name(cfg)
    cfg.MODEL.output_dir = os.path.join(args.output_dir + name)
    cfg.MODEL.name = name

    # Create output dir
    if not os.path.exists(cfg.MODEL.output_dir):
        os.makedirs(cfg.MODEL.output_dir)

    # Save config file to output dir
    shutil.copy(args.config_file, cfg.MODEL.output_dir + "/config.yaml")

    # Setup logger
    logger = setup_logger(name=cfg.MODEL.name, save_dir=cfg.MODEL.output_dir)
    dump_yaml_to_log(args.config_file, logger)

    # Random seed
    set_random_seed(cfg.SEED)

    # Dataset and Dataloader
    dataset_train, classes = build_dataset(cfg)
    logger.info(f"Dataset: Number of samples: {len(dataset_train)} | Number of classes: {classes}")
    train_loader = DataLoader(dataset_train, batch_size=cfg.DATALOADER.batch_size, shuffle=True,
                              num_workers=cfg.DATALOADER.num_workers)

    # Calculate checkpoints (iterations, save_iter, and warmup_steps based on the batch size)
    calc_training_params(cfg, len(dataset_train))

    # Device
    device = torch.device(cfg.TRAIN.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Model
    if cfg.MODEL.HEAD.name == "DynM-ArcFace":
        cls_dist = get_num_samples_per_cls(dataset_train, cfg.DATASET.cls_dist_file)
        train_model = TrainModel(cfg, classes, cls_dist)
    else:
        train_model = TrainModel(cfg, classes)

    if cfg.MODEL.BACKBONE.freeze_backbone:
        train_model.freeze_backbone()

    if args.data_parallelism:
        cfg.TRAIN.data_parallel = True
        gpu_ids = cfg.TRAIN.gpu_ids
        train_model = nn.DataParallel(train_model, device_ids=gpu_ids)
        logger.info(f"Using multiple GPU's: cuda: {gpu_ids}")
    else:
        cfg.TRAIN.data_parallel = False

    cfg.freeze()

    train_model.to(device)

    # Loss
    criterion = build_loss(cfg)

    # Optimizer
    optimizer = build_optimizer(cfg, train_model)

    # Scheduler
    scheduler = build_scheduler(cfg, optimizer)

    # Warmup scheduler
    warmup_scheduler = build_warmup_scheduler(cfg, optimizer)

    # Training loop
    train_tracker = TrainTracker(config=cfg)

    if cfg.TRAIN.epoch_based:
        train_time = epoch_based_training(cfg, train_model, criterion, optimizer, train_loader, device, scheduler,
                                          warmup_scheduler, logger, train_tracker)
    else:
        train_time = iter_based_training(cfg, train_model, criterion, optimizer, train_loader, device, scheduler,
                                         warmup_scheduler, logger, train_tracker)

    # Plot learning rate scheduler & loss
    train_tracker.plot_lrs(os.path.join(cfg.MODEL.output_dir, "lrs_scheduling.png"))
    train_tracker.plot_loss(os.path.join(cfg.MODEL.output_dir, "train_loss.png"))

    # Finished training
    logger.info(f"Total training time: {str(datetime.timedelta(seconds=int(time.time() - train_time)))}")
