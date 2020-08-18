import logging
import os
import re
import requests
import tarfile
import zipfile

from requests.models import Request, HTTPError
from tqdm import tqdm


logger = logging.getLogger(__name__)


def download(data_location, data_dir, force_download=False):
    """ Downloads a dataset from `data_location` to `data_dir`. """
    req = requests.get(data_location, stream=True)

    if req.status_code != 200:
        logger.error(f'Could not download {data_location}: {req.status_code}')
        raise HTTPError('data_location cannot be donwloaded')

    target_filename = extract_file_name(req)
    destination = os.path.join(data_dir, target_filename)

    if os.path.exists(destination) and not force_download:
        logger.warn(f'File {destination} already exists. Aborting.')
        return destination

    logger.info(f'File to download: {target_filename} from {data_location}')

    os.makedirs(data_dir, exist_ok=True)  # Assure data Dir

    total_download_size = int(req.headers.get('Content-Length', 0))
    download_block_size = 1024

    with tqdm(total=total_download_size, unit='iB', unit_scale=True) as bar:
        with open(destination, 'wb+') as datafile:
            for block in req.iter_content(download_block_size):
                datafile.write(block)
                bar.update(len(block))

    logger.info(f'File downloaded to {destination} successfully.')

    return destination


def download_and_extract(data_location, data_dir):
    downloaded_file = download(data_location, data_dir)

    if tarfile.is_tarfile(downloaded_file):
        with tarfile.open(downloaded_file, 'r') as tar:
            tar.extractall(data_dir)
    elif zipfile.is_zipfile(downloaded_file):
        with zipfile.ZipFile(downloaded_file) as zip:
            zip.extractall(data_dir)
    else:
        logger.warn(f'File {downloaded_file} is neither tar nor zip.')

    os.remove(downloaded_file)


def extract_file_name(req: Request):
    target_filename = None

    if 'content-disposition' in req.headers:
        disposition = req.headers['content-disposition']
        file_match = re.match(r'filename=\"(.+)\"', disposition)

        if file_match:
            target_filename = file_match.group(0)

    if not target_filename:
        path = req.url
        target_filename = path.split('/')[-1]

    return target_filename
