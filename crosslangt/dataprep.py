import logging
import os
import torch

from pickle import HIGHEST_PROTOCOL

from .dataset_utils import download

from transformers.data import (SquadV1Processor,
                               squad_convert_examples_to_features)

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
