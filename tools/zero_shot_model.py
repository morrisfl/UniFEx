import argparse
import logging
import os.path
from zipfile import ZipFile

import torch

from model import InferenceModel
from train_pipeline import set_random_seed
from utils import get_cfg_defaults, google_drive_upload


def pars_args():
    parser = argparse.ArgumentParser(description="Construct zero-shot model.")
    parser.add_argument("config_file", help="Path to the config file")

    return parser.parse_args()


if __name__ == "__main__":
    args = pars_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.output_dir = os.path.join("results", "zero_shot_models")

    work_dir = os.getcwd()
    if os.path.basename(work_dir) == "tools":
        cfg.MODEL.output_dir = "../results/zero_shot_models"
        cfg.GOOGLE_DRIVE.client_secret_path = "../credentials/client_secret.json"
        cfg.GOOGLE_DRIVE.token_path = "../credentials/token_drive_v3.pickle"

    if not os.path.exists(cfg.MODEL.output_dir):
        os.makedirs(cfg.MODEL.output_dir)

    logger = logging.getLogger(f"Zero-shot model {cfg.MODEL.name}")
    logger.setLevel(logging.INFO)

    set_random_seed(cfg.SEED)

    model = InferenceModel(cfg)
    model.eval()

    x = torch.rand(1, 3, model.backbone.img_size, model.backbone.img_size)
    y = model(x)
    logger.info(f"Input shape: {x.shape} | Output shape: {y.shape}")

    save_path = os.path.join(cfg.MODEL.output_dir, cfg.MODEL.name + ".pt")
    zip_path = os.path.join(cfg.MODEL.output_dir, cfg.MODEL.name + ".zip")

    saved_model = torch.jit.script(model)
    saved_model.save(save_path)
    logger.info(f"Model saved to {save_path}")

    with ZipFile(zip_path, "w") as zip_file:
        zip_file.write(save_path, arcname=os.path.basename(save_path))
        zip_file.write(args.config_file, arcname="config.yaml")

    try:
        google_drive_upload(cfg, zip_path, logger)
        os.remove(zip_path)
        os.remove(save_path)
    except Exception as e:
        logger.info(f"Unable to upload file {os.path.basename(zip_path)} to Google Drive!")
        logger.info(e)
