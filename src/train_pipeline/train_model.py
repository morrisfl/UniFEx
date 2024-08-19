import datetime
import os.path
import random
import time
from zipfile import ZipFile

import numpy as np
import torch
from tqdm import tqdm

from model import get_inference_model
from utils.google_drive import google_drive_upload
from train_pipeline.optimizer import SAM


def epoch_based_training(config, model, criterion, optimizer, data_loader, device, scheduler, warmup_scheduler, logger,
                         tracker):

    start_time = time.time()
    for epoch in range(1, config.TRAIN.epochs + 1):
        logger.info(f"---------------------- Epoch: {epoch}/{config.TRAIN.epochs} -----------------------")
        epoch_loss = _train_one_epoch(config, model, criterion, optimizer, data_loader, device, scheduler,
                                      warmup_scheduler, logger, tracker)

        tracker.reset_epoch(epoch_loss, epoch + 1)

        if config.SCHEDULER.epoch_based and scheduler is not None:
            if warmup_scheduler is not None:
                with warmup_scheduler.dampening():
                    scheduler.step()
            else:
                scheduler.step()

        logger.info(f"Epoch: {epoch}/{config.TRAIN.epochs} | Time Spend: "
                    f"{str(datetime.timedelta(seconds=int(time.time() - start_time)))} | Train Loss: {epoch_loss:.4f}")

        if epoch in config.TRAIN.save_epoch:
            model_name = config.MODEL.name + f"_epoch{epoch}"
            save_inf_model(config, model, model_name, logger)

    return start_time


def iter_based_training(config, model, criterion, optimizer, data_loader, device, scheduler, warmup_scheduler, logger,
                        tracker):

    start_time = time.time()
    logger.info(f"-------------------------- Iteration to train: {config.TRAIN.iterations} ---------------------------")
    data_iter = iter(data_loader)
    model.train()

    for i in tqdm(range(1, config.TRAIN.iterations + 1), desc=f"Train {config.TRAIN.iterations} iterations"):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            images, labels = next(data_iter)

        batch_loss = _train_one_batch(config, model, criterion, optimizer, images, labels, device, scheduler,
                                      warmup_scheduler, tracker)

        tracker.running_iter_loss += batch_loss
        tracker.running_loss += batch_loss

        if i % config.TRAIN.save_iter == 0:
            model_name = config.MODEL.name + f"_iter{tracker.curr_iter}"
            save_inf_model(config, model, model_name, logger)
            model.train()
            tracker.reset_iter()

        if i % config.TRAIN.print_freq == 0:
            curr_avg_loss = tracker.running_loss / config.TRAIN.print_freq
            tracker.running_loss = 0.0
            logger.info(f"Training Iteration: {i}/{config.TRAIN.iterations} | Loss: {curr_avg_loss:.4f}")

    return start_time


def _train_one_epoch(config, model, criterion, optimizer, data_loader, device, scheduler, warmup_scheduler, logger,
                     tracker):
    model.train()
    for images, labels in tqdm(data_loader, desc=f"Train epoch {tracker.curr_epoch}/{config.TRAIN.epochs}"):

        batch_loss = _train_one_batch(config, model, criterion, optimizer, images, labels, device, scheduler,
                                      warmup_scheduler, tracker)

        tracker.running_loss += batch_loss
        tracker.running_epoch_loss += batch_loss
        tracker.curr_iter += 1

        if tracker.curr_iter % config.TRAIN.save_iter == 0 and not config.TRAIN.epoch_based:
            model_name = config.MODEL.name + f"_iter{tracker.curr_iter}"
            save_inf_model(config, model, model_name, logger)
            model.train()

        if tracker.curr_iter % config.TRAIN.print_freq == 0:
            curr_avg_loss = tracker.running_loss / config.TRAIN.print_freq
            tracker.running_loss = 0.0
            logger.info(f"Training Epoch: {tracker.curr_epoch}/{config.TRAIN.epochs} | "
                        f"Step: [{tracker.curr_iter}/{len(data_loader)}] | Loss: {curr_avg_loss:.4f}")

    return tracker.running_epoch_loss / len(data_loader)


def _train_one_batch(config, model, criterion, optimizer, images, labels, device, scheduler, warmup_scheduler, tracker):
    images = images.to(device)
    labels = labels.to(device)

    if isinstance(optimizer, SAM):
        # first forward-backward pass
        logits = model(images, labels)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.first_step(zero_grad=True)

        # second forward-backward pass
        criterion(model(images, labels), labels).backward()
        optimizer.second_step(zero_grad=True)
    else:
        logits = model(images, labels)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if not config.SCHEDULER.epoch_based:
        if scheduler is not None:
            if warmup_scheduler is not None:
                with warmup_scheduler.dampening():
                    scheduler.step()
            else:
                scheduler.step()
    else:
        if warmup_scheduler is not None:
            with warmup_scheduler.dampening():
                pass

    tracker.lrs.append(optimizer.param_groups[0]['lr'])

    return loss.item()


def save_inf_model(config, model, model_name, logger):
    inf_model = get_inference_model(config, model)
    inf_model.to("cpu")
    inf_model.eval()

    path = os.path.join(config.MODEL.output_dir, model_name + ".pt")
    zip_path = os.path.join(config.MODEL.output_dir, model_name + ".zip")

    saved_model = torch.jit.script(inf_model)
    saved_model.save(path)
    logger.info(f"Saved model to {path}")

    try:
        loaded_model = torch.jit.load(path, map_location="cpu")
        loaded_model.eval()
        x = torch.randn(1, 3, 224, 224).to("cpu")
        y = loaded_model(x)
        logger.info(f"Model loaded successfully. Output shape: {y.size()}")
    except Exception as e:
        logger.info(f"Model loading failed. {e}")
        # os.remove(path)
        # logger.info(f"Removed model from {path}")

    if config.MODEL.cloud_upload and os.path.exists(path):
        with ZipFile(zip_path, "w") as zip_file:
            zip_file.write(path, arcname=os.path.basename(path))
            zip_file.write(os.path.join(config.MODEL.output_dir, "config.yaml"), arcname="config.yaml")

        try:
            google_drive_upload(config, zip_path, logger)
            logger.info(f"Model {model_name} uploaded to Google Drive!")
            os.remove(zip_path)
            os.remove(path)
        except Exception as e:
            logger.info(f"Unable to upload file {os.path.basename(path)} to Google Drive!")
            logger.info(e)


def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)
