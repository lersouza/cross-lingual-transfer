from multiprocessing import process
import os
from pickle import HIGHEST_PROTOCOL
from typing import List
import pytorch_lightning as pl

from dataclasses import dataclass
import torch
from crosslangt.dataset_utils import download

from transformers import AutoTokenizer
from transformers.data.processors.utils import DataProcessor
from transformers.data.processors.squad import (
    SquadExample, SquadFeatures, SquadProcessor, SquadResult, SquadV1Processor,
    squad_convert_examples_to_features)


@dataclass
class DataConfig:
    name: str
    train_url: str
    eval_url: str
    test_url: str
    processor: DataProcessor


class FaquadProcessor(SquadProcessor):
    train_file = "train.json"
    dev_file = "dev.json"


class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, examples: List[SquadExample],
                 features: List[SquadFeatures]) -> None:

        self.examples = examples
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]

        item = {
            'input_ids': feature.input_ids,
            'attention_mask': feature.attention_mask,
            'feature_id': feature.unique_id
        }

        if feature.start_position is not None:
            item['start_positions'] = feature.start_position
            item['end_positions'] = feature.end_position

        if feature.token_type_ids is not None:
            item['token_type_ids'] = feature.token_type_ids


class SquadDataModule(pl.LightningDataModule):
    DATASETS = {
        'squad_en': {
            'train': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
            'train-v1.1.json',
            'eval': 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
            'dev-v1.1.json',
            'processor': SquadV1Processor()
        },
        'faquad': {
            'train': 'https://raw.githubusercontent.com/liafacom/faquad/master'
            '/data/train.json',
            'eval': 'https://raw.githubusercontent.com/liafacom/faquad/master/'
            'data/dev.json',
            'processor': FaquadProcessor(),
        },
        'squad_pt': {
            'train': 'https://raw.githubusercontent.com/nunorc/squad-v1.1-pt/'
            'master/train-v1.1-pt.json',
            'eval': 'https://raw.githubusercontent.com/nunorc/squad-v1.1-pt/'
            'master/dev-v1.1-pt.json',
            'processor': SquadV1Processor()
        }
    }

    def __init__(self,
                 dataset_name: str,
                 tokenizer_name: str,
                 data_dir: str,
                 batch_size: int,
                 max_seq_length: int,
                 max_query_length: int,
                 doc_stride: int,
                 data_key: str = None,
                 eval_split: str = 'eval',
                 test_split: str = 'eval') -> None:

        super().__init__()

        self.data_config = self.DATASETS[dataset_name]
        self.eval_split = eval_split
        self.test_split = test_split

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length
        self.doc_stride = doc_stride

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer_name = tokenizer_name
        self.data_key = data_key

    def prepare_data(self):
        train_location = download(self.data_config['train'], self.data_dir)
        eval_location = download(self.data_config[self.eval_split],
                                 self.data_dir)
        test_location = download(self.data_config[self.test_split],
                                 self.data_dir)

        self._process_dataset(train_location, 'train')
        self._process_dataset(eval_location, self.eval_split)
        self._process_dataset(test_location, self.test_split)

    def setup(self, stage):
        if stage == 'fit':
            train_objects = torch.load(self._gen_dataset_filename('train'))
            eval_objects = torch.load(
                self._gen_dataset_filename(self.eval_split))

            self.train_dataset = SquadDataset(train_objects[0],
                                              train_objects[1])
            self.eval_dataset = SquadDataset(eval_objects[0], eval_objects[1])
        elif stage == 'test':
            test_objects = torch.load(
                self._gen_dataset_filename(self.test_split))

            self.test_dataset = SquadDataset(test_objects[0], test_objects[1])

    def retrieve_examples_and_features(self,
                                       dataset: str = 'eval',
                                       results: List[SquadResult] = None):
        dataset = (self.test_dataset
                   if dataset == 'test' else self.eval_dataset)

        examples = dataset.examples
        features = dataset.features

        if results is not None and len(results) != len(features):
            eval_features_index = {f.unique_id: f for f in features}

            # Working with a subset of the data (probably Fast Dev Run Mode)
            examples = [
                examples[eval_features_index[i.unique_id].example_index]
                for i in results
            ]

            features = [eval_features_index[i.unique_id] for i in results]

        return examples, features

    def _process_dataset(self, file_location: str, split: str):

        processor = self.data_config.processor
        examples = (processor.get_train_examples(self.data_dir) if split
                    == 'train' else processor.get_dev_examples(self.data_dir))

        features = squad_convert_examples_to_features(
            examples,
            self.tokenizer,
            self.max_seq_length,
            self.doc_stride,
            self.max_query_length,
            split == 'train',
            return_dataset=False,
        )

        file_name = self._gen_dataset_filename(split)
        file_path = os.path.join(self.data_dir, file_name)

        torch.save((examples, features),
                   file_path,
                   pickle_protocol=HIGHEST_PROTOCOL)

    def _gen_dataset_filename(self, split: str):
        suffix = f'-{self.data_key}' if self.data_key else ''

        return (f'{self.data_config.name}-{split}-{self.tokenizer_name}'
                f'-{self.max_seq_length}{suffix}.ds')
