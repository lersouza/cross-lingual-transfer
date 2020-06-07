import argparse
import logging
import os.path
import requests

from tqdm import tqdm


WIKI_LATEST='https://dumps.wikimedia.org/ptwiki/latest/' \
            'ptwiki-latest-pages-articles.xml.bz2'


logger = logging.getLogger(__name__)


def download_pt_wiki_dump(dest_location,
                          src_url=WIKI_LATEST):
    """ 
    Downloads the Wiki-Dump for pre-training.

    dest_location: The destination directory where the dump will be saved to.
    src_url: Wiki Dump Location. Default is the latest ptwiki dump.
    """
    logger.info(f'Downloading Wiki Dump from {src_url} to {dest_location}')

    if not dest_location or not os.path.isdir(dest_location):
        raise ValueError(
            f'Destination must be a directory (got {dest_location})')

    req = requests.get(src_url, stream=True)
    total_download_size = int(req.headers.get('Content-Length', 0))
    download_block_size = 1024
    
    destination_file = os.path.join(dest_location, src_url.split('/')[-1])

    if os.path.exists(destination_file):
        os.remove(destination_file)

    with tqdm(total=total_download_size, unit='iB', unit_scale=True) as bar:
        with open(destination_file, 'wb+') as wiki_dump:
            for block in req.iter_content(download_block_size):
                wiki_dump.write(block)
                bar.update(len(block))

    logger.info('Wiki Dump download successfully.')

   
def main():
    """ Run this program. """
    parser = argparse.ArgumentParser()

    parser.add_argument('dest_dir', default='./',
                        help='Destination directory to save dump file.')
    parser.add_argument('--wiki_dump_url', default=WIKI_LATEST)

    args = parser.parse_args()

    download_pt_wiki_dump(args.dest_dir, args.wiki_dump_url)


if __name__ == '__main__':
    main()
