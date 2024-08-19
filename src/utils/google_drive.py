import io
import pickle
import os

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.auth.transport.requests import Request


def create_service(config):
    client_secret_file = config.GOOGLE_DRIVE.client_secret_path
    api_name = config.GOOGLE_DRIVE.api_name
    api_version = config.GOOGLE_DRIVE.api_version
    scope = config.GOOGLE_DRIVE.scope
    token_path = config.GOOGLE_DRIVE.token_path

    cred = None

    if os.path.exists(token_path):
        with open(token_path, 'rb') as token:
            cred = pickle.load(token)

    if not cred or not cred.valid:
        if cred and cred.expired and cred.refresh_token:
            try:
                cred.refresh(Request())
            except Exception:
                os.remove(token_path)
                cred = _auth_google_drive(client_secret_file, scope)

        else:
            cred = _auth_google_drive(client_secret_file, scope)

        with open(token_path, 'wb') as token:
            pickle.dump(cred, token)

    try:
        service = build(api_name, api_version, credentials=cred)
        return service

    except Exception as e:
        print(e)
        return None


def _auth_google_drive(secret, scope):
    flow = InstalledAppFlow.from_client_secrets_file(secret, scope)
    return flow.run_local_server()


def google_drive_upload(config, file_path, logger, mime_type='application/zip'):
    service = create_service(config)

    file_name = os.path.basename(file_path)
    file_metadata = {
        'name': file_name,
        'parents': [config.GOOGLE_DRIVE.folder_id]
    }

    media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)

    service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()

    logger.info(f"Uploaded file '{file_name}' to Google Drive")


def google_drive_update(config, file_path, logger,
                        mime_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'):
    service = create_service(config)

    file_name = os.path.basename(file_path)
    file_id = get_file_id(config, file_name, logger)

    media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)

    service.files().update(
        fileId=file_id,
        media_body=media,
    ).execute()

    logger.info(f"Updated file '{file_name}' on Google Drive")


def google_drive_download(config, file_name, save_path, logger):
    try:
        service = create_service(config)

        file_id = get_file_id(config, file_name, logger)

        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False

        while done is False:
            status, done = downloader.next_chunk()
            # logger.info(f"Downloaded {int(status.progress() * 100)}%.")

        fh.seek(0)
        with open(os.path.join(save_path, file_name), 'wb') as f:
            f.write(fh.read())
            f.close()

        logger.info(f"Downloaded file '{file_name}' from Google Drive.")

    except HttpError as e:
        logger.warning(f"Unable to download file '{file_name}' from Google Drive.")
        logger.warning(e)


def get_file_id(config, file_name, logger):
    try:
        service = create_service(config)

        page_token = None

        while True:
            response = service.files().list(q=f"name='{file_name}'",
                                            spaces='drive',
                                            fields='nextPageToken, files(id, name)',
                                            pageToken=page_token).execute()

            if len(response.get('files', [])) > 0:
                file = response.get('files', [])[0]
                return file.get('id')

            page_token = response.get('nextPageToken', None)

            if page_token is None:
                break

    except HttpError as e:
        logger.warning(f"Unable to find file '{file_name}' in Google Drive.")
        logger.warning(e)


if __name__ == '__main__':
    from src.utils.default_cfg import get_cfg_defaults

    cfg = get_cfg_defaults()
    cfg.GOOGLE_DRIVE.client_secret_path = "../../credentials/client_secret.json"
    cfg.GOOGLE_DRIVE.token_path = "../../credentials/token_drive_v3.pickle"

    create_service(cfg)


