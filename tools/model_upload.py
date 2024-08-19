import argparse
import logging
import os

from utils import google_drive_upload, get_cfg_defaults


def pars_args():
    parser = argparse.ArgumentParser(description="Upload models to Google Drive from a given directory.")
    parser.add_argument("result_dir", help="Path to the result directory which contains the models for upload")
    return parser.parse_args()


if __name__ == '__main__':
    args = pars_args()

    cfg = get_cfg_defaults()

    work_dir = os.getcwd()
    if os.path.basename(work_dir) == "tools":
        cfg.GOOGLE_DRIVE.client_secret_path = "../credentials/client_secret.json"
        cfg.GOOGLE_DRIVE.token_path = "../credentials/token_drive_v3.pickle"

    logger = logging.getLogger(f"Model upload from training {os.path.basename(args.result_dir)}")
    logger.setLevel(logging.INFO)

    for file in os.listdir(args.result_dir):
        if file.endswith(".zip"):
            try:
                path = os.path.join(args.result_dir, file)
                google_drive_upload(cfg, path, logger)
                os.remove(path)
            except Exception as e:
                logger.info(e)
