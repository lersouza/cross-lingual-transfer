import logging
import os
from pickle import HIGHEST_PROTOCOL
import re
import tarfile
import torch

import requests
from requests.models import HTTPError, Request
from tqdm import tqdm
from transformers.data import (SquadV1Processor,
                               squad_convert_examples_to_features)
from transformers.data.processors.squad import SquadExample, SquadFeatures
from transformers.tokenization_bert import BertTokenizer

logger = logging.getLogger(__name__)


QA_DATASETS = {
    'faquad': {
        'train': 'https://raw.githubusercontent.com/liafacom/faquad/master'
                 '/data/train.json',
        'eval': 'https://raw.githubusercontent.com/liafacom/faquad/master/'
                'data/dev.json'
    },
    'squad-en': {
        'train': '',
        'eval': '',
    },
    'squad-pt': {
        'train': '',
        'eval': '',
    },
}


def prepare_question_answer_dataset(dataset: str,
                                    split: str,
                                    data_dir: str,
                                    tokenizer: BertTokenizer,
                                    max_seq_length: int,
                                    doc_stride: int,
                                    max_query_length: int):
    """
    Downloads and prepare a Question Answering Dataset.
    """
    assert dataset in QA_DATASETS
    assert split in ['train', 'eval']
    assert data_dir is not None
    assert tokenizer is not None
    assert max_seq_length > 0
    assert doc_stride > 0
    assert max_query_length > 0

    # Prepare the Data
    final_dataset_filename = f'{dataset}-{split}-{max_seq_length}.dataset'
    final_dataset_path = os.path.join(data_dir, final_dataset_filename)

    if os.path.exists(final_dataset_path):
        logger.info(f'Dataset {final_dataset_path} already there. Skipping.')
        return  # File are already there!

    logger.info(f'Dataset will be generated at: {final_dataset_path}.')

    dataset_location = QA_DATASETS[dataset][split]
    original_dataset = download(dataset_location, data_dir)

    processor = SquadV1Processor()

    _, original_filename = os.path.split(original_dataset)

    if split == 'train':
        examples = processor.get_train_examples(data_dir, original_filename)
    else:
        examples = processor.get_dev_examples(data_dir, original_filename)

    features, tensor_dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=split == 'train',
        return_dataset="pt",
        threads=os.cpu_count(),
    )

    with open(final_dataset_path, 'wb+') as dot_dataset:
        torch.save({
            'examples': examples,
            'features': features,
            'dataset': tensor_dataset
        }, dot_dataset, pickle_protocol=HIGHEST_PROTOCOL)


def load_question_answer_dataset(dataset: str,
                                 split: str,
                                 data_dir: str,
                                 max_seq_length: int):
    dataset_filename = f'{dataset}-{split}-{max_seq_length}.dataset'
    dataset_filepath = os.path.join(data_dir, dataset_filename)

    if not os.path.exists(dataset_filepath):
        logger.error(f'Dataset {dataset_filepath} not found. Consider prepare it first!')
        raise IOError(f'Dataset {dataset_filepath} was not found.')

    with open(dataset_filepath, 'rb') as dataset:
        contents = torch.load(dataset)

    return contents


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

    with tarfile.open(downloaded_file, 'r') as tar:
        tar.extractall(data_dir)

    os.remove(downloaded_file)


def extract_file_name(req: Request):
    target_filename = None

    if 'content-disposition' in req.headers:
        disposition = req.headers['content-disposition']
        target_filename = re.findall('filename=\"(.+)\"', disposition)[0]
    else:
        path = req.url
        target_filename = path.split('/')[-1]

    return target_filename


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    prepare_question_answer_dataset(
        'farquad',
        'train',
        '/tmp/farquad/',
        BertTokenizer.from_pretrained('bert-base-cased'),
        max_seq_length=384,
        max_query_length=64,
        doc_stride=128)
