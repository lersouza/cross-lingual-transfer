import argparse
import logging
import os.path
import requests

from tqdm import tqdm


logger = logging.getLogger(__name__)


def download_wiki_dump(language, destination_dir, force_download=False):
    """
    Downloads the Wiki-Dump for pre-training.

    language: The Wiki Dump language to download.
    destination_dir: The destination directory where the dump will be saved to.
    force_download: Indicates whether to force download even if file exists.
    """
    assert not os.path.exists(destination_dir) \
        or os.path.isdir(destination_dir)

    wiki_file = f'{language}wiki-latest-pages-articles.xml.bz2'
    src_url = f'https://dumps.wikimedia.org/{language}wiki/latest/{wiki_file}'
    target_file = os.path.join(destination_dir, wiki_file)

    if os.path.exists(target_file) and not force_download:
        logger.info(f'File {target_file} already exists. Skipping...')
        return

    logger.info(f'Downloading Wiki Dump from {src_url} to {destination_dir}')

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir, exist_ok=True)

    req = requests.get(src_url, stream=True)

    if req.status_code != 200:
        logging.info(f'Could not download file from {src_url}. '
                     f'Status was: {req.status_code}')

    total_download_size = int(req.headers.get('Content-Length', 0))
    download_block_size = 1024

    if os.path.exists(target_file):
        os.remove(target_file)

    with tqdm(total=total_download_size, unit='iB', unit_scale=True) as bar:
        with open(target_file, 'wb+') as wiki_dump:
            for block in req.iter_content(download_block_size):
                wiki_dump.write(block)
                bar.update(len(block))

    logger.info('Wiki Dump download successfully.')


def main():
    """ Run this program. """
    parser = argparse.ArgumentParser()

    parser.add_argument('lang', help='The Wiki Dump language to download.')
    parser.add_argument('dest_dir', default='./',
                        help='Destination directory to save dump file.')

    parser.add_argument('--force', action='store_true',
                        help='Indicates whether to force the download '
                             'even if the file exists')

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    download_wiki_dump(args.lang, args.dest_dir, args.force)


if __name__ == '__main__':
    main()
