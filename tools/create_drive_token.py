import logging
import os
import pickle

from google_auth_oauthlib.flow import InstalledAppFlow

from utils import get_cfg_defaults


def update_google_token():
    """Update Google Drive authorization token (token expires after 7 days)."""

    cfg = get_cfg_defaults()

    work_dir = os.getcwd()
    if os.path.basename(work_dir) == "tools":
        cfg.GOOGLE_DRIVE.client_secret_path = "../credentials/client_secret.json"
        cfg.GOOGLE_DRIVE.token_path = "../credentials/token_drive_v3.pickle"

    logger = logging.getLogger("Update Google Drive authorization")
    logger.setLevel(logging.INFO)

    if os.path.exists(cfg.GOOGLE_DRIVE.token_path):
        os.remove(cfg.GOOGLE_DRIVE.token_path)

    flow = InstalledAppFlow.from_client_secrets_file(cfg.GOOGLE_DRIVE.client_secret_path, cfg.GOOGLE_DRIVE.scope)
    cred = flow.run_local_server()

    with open(cfg.GOOGLE_DRIVE.token_path, "wb") as token:
        pickle.dump(cred, token)

    logger.info("Successfully created and saved new token file.")


if __name__ == '__main__':
    update_google_token()
