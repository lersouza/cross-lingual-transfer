import logging
import os
import re
import tarfile
import zipfile
from pickle import HIGHEST_PROTOCOL
from typing import List

import requests
import torch
from lxml import etree
from requests.models import HTTPError, Request
from tqdm import tqdm
from transformers.data import (DataProcessor, SquadV1Processor,
                               squad_convert_examples_to_features)
from transformers.data.processors.glue import MnliProcessor
from transformers.data.processors.utils import InputExample
from transformers.tokenization_bert import BertTokenizer

from crosslangt.datasets import NLIDataset, NLIExample


logger = logging.getLogger(__name__)


class ASSIN2Processor(DataProcessor):
    """ Converts ASSIN2 dataset files into InputExample objects. """

    def get_train_examples(self, data_dir):
        return self._create_examples(
            os.path.join(data_dir, 'assin2-train-only.xml'),
            'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            os.path.join(data_dir, 'assin2-test.xml'),
            'dev')

    def _create_examples(self, file, set_type) -> List[InputExample]:
        with open(file, 'r') as assin_xml:
            xml_file_contents = assin_xml.read()
            xml_file_contents = xml_file_contents.encode('utf-8')

        root = etree.fromstring(xml_file_contents)
        examples = []

        for i, pair in enumerate(root):
            entailment = pair.attrib['entailment'].lower()
            pairID = pair.attrib['id']
            sentence1 = sentence2 = ''

            if entailment not in ['none', 'entailment']:
                continue  # Ignoring labels that are not in MNLI dataset

            for child in pair:
                if child.tag == 't':
                    sentence1 = child.text
                else:
                    sentence2 = child.text

            examples.append(InputExample(
                pairID, sentence1, sentence2, entailment
            ))

        return examples


QA_DATASETS = {
    'faquad': {
        'train': 'https://raw.githubusercontent.com/liafacom/faquad/master'
                 '/data/train.json',
        'eval': 'https://raw.githubusercontent.com/liafacom/faquad/master/'
                'data/dev.json'
    },
    'squad-en': {
        'train': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
                 'train-v1.1.json',
        'eval': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
                'dev-v1.1.json',
    },
    'squad-pt': {
        'train': 'https://raw.githubusercontent.com/nunorc/squad-v1.1-pt/'
                 'master/train-v1.1-pt.json',
        'eval': 'https://raw.githubusercontent.com/nunorc/squad-v1.1-pt/'
                 'master/dev-v1.1-pt.json',
    },
}


# In the NLI Datasets, the labels must be aligned, so transfer learning
# across languages is possible.
NLI_DATASETS = {
    'mnli': {
        'zip': 'https://firebasestorage.googleapis.com/v0/b/'
               'mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip'
               '?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce',
        'train': 'MNLI/train.tsv',
        'eval': 'MNLI/dev_matched.tsv',
        'processor': MnliProcessor,
        'labels': ['neutral', 'entailment', 'contradiction']
    },
    'assin2': {
        'zip': 'https://github.com/lersouza/cross-lingual-transfer/raw/'
               'master/datasets/ASSIN2.zip',
        'train': 'ASSIN2/assin2-train-only.xml',
        'eval': 'ASSIN2/assin2-test.xml',
        'processor': ASSIN2Processor,
        'labels': ['none', 'entailment']
    }
}


def prepare_nli_dataset(dataset: str,
                        split: str,
                        data_dir: str,
                        tokenizer: BertTokenizer,
                        max_seq_length: int):
    assert dataset in NLI_DATASETS
    assert split in ['train', 'eval']
    assert data_dir is not None

    final_dataset_filename = f'nli-{dataset}-{split}-{max_seq_length}.dataset'
    final_dataset_path = os.path.join(data_dir, final_dataset_filename)

    if os.path.exists(final_dataset_path):
        logger.info(f'Dataset {final_dataset_path} already there. Skipping.')
        return  # File are already there!

    logger.info(f'Dataset will be generated at: {final_dataset_path}.')

    data_config = NLI_DATASETS[dataset]
    data_file_path = os.path.join(data_dir, data_config[split])

    if not os.path.exists(data_file_path):
        logger.info(f'Downloading zip file for {data_file_path}.')
        download_and_extract(data_config['zip'], data_dir)

    processor = data_config['processor']()
    nli_base_dir, _ = os.path.split(data_file_path)

    logger.info(f'About to extract examples in {nli_base_dir} '
                f'with {type(processor)}')

    if split == 'train':
        examples = processor.get_train_examples(nli_base_dir)
    else:
        examples = processor.get_dev_examples(nli_base_dir)

    features = []
    available_labels = data_config['labels']

    for example in tqdm(examples, desc='tokenizing examples'):
        encoded = tokenizer.encode_plus(
            example.text_a, example.text_b,
            max_length=max_seq_length, pad_to_max_length=True)

        features.append(NLIExample(
            encoded['input_ids'],
            encoded['attention_mask'],
            encoded['token_type_ids'],
            available_labels.index(example.label),
            example.guid
        ))

    with open(final_dataset_path, 'wb') as processed_dataset_file:
        torch.save(features, processed_dataset_file,
                   pickle_protocol=HIGHEST_PROTOCOL)


def load_nli_dataset(dataset: str,
                     split: str,
                     data_dir: str,
                     max_seq_length: int):
    dataset_filename = f'nli-{dataset}-{split}-{max_seq_length}.dataset'
    dataset_filepath = os.path.join(data_dir, dataset_filename)

    if not os.path.exists(dataset_filepath):
        logger.error(f'Dataset {dataset_filepath} not found. '
                     'Consider prepare first!')

        raise IOError(f'Dataset {dataset_filepath} was not found.')

    with open(dataset_filepath, 'rb') as dataset:
        contents = torch.load(dataset)

    return NLIDataset(contents)


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
        logger.error(f'Dataset {dataset_filepath} not found. '
                     'Consider prepare first!')

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


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    prepare_nli_dataset(
        'assin2',
        'eval',
        '/tmp/assin2/',
        tokenizer,
        max_seq_length=128)

    dataset = load_nli_dataset(
        'assin2',
        'eval',
        '/tmp/assin2',
        128
    )

    print(len(dataset))
    print(tokenizer.decode(dataset[0]['input_ids']))
