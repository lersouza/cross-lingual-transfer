import logging
import os
from sys import path

import pytorch_lightning as pl
import torch

from dataclasses import dataclass
from pickle import HIGHEST_PROTOCOL
from typing import Dict, List, Optional
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer

from deeppavlov.dataset_readers.squad_dataset_reader import SquadDatasetReader
from deeppavlov.dataset_iterators.squad_iterator import SquadIterator
from deeppavlov.models.preprocessors.squad_preprocessor import (
    SquadBertAnsPostprocessor, SquadBertMappingPreprocessor,
    SquadBertAnsPreprocessor)
from transformers.data.processors.squad import SquadResult

logger = logging.getLogger('data_qa')


@dataclass
class SquadInputFeatures:
    original_index: int
    tokens: List[str]
    input_ids: List[int]
    input_mask: List[int]
    input_type_ids: Optional[List[int]]


KNWON_QA_DATASETS = {
    'squad_en': {
        'url': None,
        'original_dataset_name': 'SQuAD'
    },
    'faquad': {
        'url': 'https://raw.githubusercontent.com/lersouza/'
        'cross-lingual-transfer/master/datasets/faquad.tar.gz',
        'original_dataset_name': 'FaQuAD'
    },
    'drcd': {
        'url': 'http://files.deeppavlov.ai/datasets/DRCD.tar.gz',
        'original_dataset_name': 'DRCD'
    },
    'sbersquad': {
        'url':
        'http://files.deeppavlov.ai/datasets/sber_squad_clean-v1.1.tar.gz',
        'original_dataset_name': 'SberSQuADClean',
    }
}


class SquadDataset(Dataset):
    def __init__(self, input_features, contexts_raw, answers, answers_start,
                 answers_end, tok2char, char2tok):

        self.input_features = input_features
        self.contexts_raw = contexts_raw
        self.answers = answers
        self.answers_start = answers_start
        self.answers_end = answers_end
        self.tok2char = tok2char
        self.char2tok = char2tok

    def __len__(self) -> int:
        return len(self.input_features)

    def __getitem__(self, index: int):
        features = self.input_features[index]
        ans_start = self.answers_start[index]
        ans_end = self.answers_end[index]

        example = {
            'index': torch.tensor(features.original_index).long(),
            'input_ids': torch.tensor(features.input_ids).long(),
            'attention_mask': torch.tensor(features.input_mask).long(),
            'token_type_ids': torch.tensor(features.input_type_ids).long(),
            'start_positions': torch.tensor(ans_start[0]).long(),
            'end_positions': torch.tensor(ans_end[0]).long()
        }

        return example


class SquadDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_name: str,
                 tokenizer_name: str,
                 data_dir: str,
                 batch_size: int,
                 max_seq_length: int,
                 max_query_length: int,
                 data_key: str = None,
                 eval_split: str = 'eval',
                 test_split: str = 'eval',
                 preprocess_threads: int = 1,
                 dataset_custom_config: Dict[str, Dict] = None,
                 do_lower_case: bool = False) -> None:

        super().__init__()

        self.dataset_name = dataset_name
        self.data_config = (dataset_custom_config
                            or KNWON_QA_DATASETS[dataset_name])

        # Using DeepPavlov split names instead of eval
        self.train_split = 'train'
        self.eval_split = 'valid' if eval_split == 'eval' else eval_split
        self.test_split = 'valid' if test_split == 'eval' else test_split

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.max_query_length = max_query_length

        self.preprocess_threads = preprocess_threads

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, do_lower_case=do_lower_case)

        self.tokenizer_name = tokenizer_name
        self.do_lower_case = do_lower_case
        self.data_key = data_key

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=8,
                          shuffle=True)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.eval_dataset,
                          batch_size=self.batch_size,
                          num_workers=8,
                          shuffle=False)

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=8,
                          shuffle=False)

    def prepare_data(self):
        dataset_reader = SquadDatasetReader()
        dataset = dataset_reader.read(
            os.path.join(self.data_dir, self.dataset_name),
            self.data_config['original_dataset_name'],
            url=self.data_config['url'])

        iterator = SquadIterator(dataset)
        bert_mappings = SquadBertMappingPreprocessor(
            do_lower_case=self.do_lower_case)
        ans_preprocessor = SquadBertAnsPreprocessor(
            do_lower_case=self.do_lower_case)

        for split in [self.train_split, self.eval_split, self.test_split]:
            target_dataset_file = self._gen_dataset_filename(split)

            if os.path.exists(target_dataset_file):
                logger.info(f'Dataset file '{target_dataset_file}'
                            'already exists. Skipping.')
                continue

            instances = iterator.get_instances(split)

            contexts_raw = []
            answers_raw = []
            answers_raw_start = []
            input_features = []

            for i, (context,
                    question) in tqdm(enumerate(instances[0]),
                                      total=len(instances[0]),
                                      desc=f'tokenizing {split} instances'):

                tokenized = self.tokenizer(question,
                                           context,
                                           truncation=True,
                                           max_length=self.max_seq_length)
                tokens = self.tokenizer.convert_ids_to_tokens(
                    tokenized['input_ids'])

                tokenized = self.tokenizer.pad(tokenized,
                                               padding='max_length',
                                               max_length=self.max_seq_length)
                in_features = SquadInputFeatures(
                    i, tokens,
                    tokenized['input_ids'], tokenized['attention_mask'],
                    tokenized.get('token_type_ids'))

                contexts_raw.append(context)
                input_features.append(in_features)

                answers_raw.append(instances[1][i][0])
                answers_raw_start.append(instances[1][i][1])

            subtok2chars, char2subtoks = bert_mappings(contexts_raw,
                                                       input_features)
            ans, ans_start, ans_end = ans_preprocessor(answers_raw,
                                                       answers_raw_start,
                                                       char2subtoks)

            torch.save(
                {
                    'input_features': input_features,
                    'contexts_raw': contexts_raw,
                    'answers': ans,
                    'answers_start': ans_start,
                    'answers_end': ans_end,
                    'tok2char': subtok2chars,
                    'char2tok': char2subtoks
                },
                self._gen_dataset_filename(split),
                pickle_protocol=HIGHEST_PROTOCOL)

    def setup(self, stage):
        if stage == 'fit':
            train_objects = torch.load(
                self._gen_dataset_filename(self.train_split))
            eval_objects = torch.load(
                self._gen_dataset_filename(self.eval_split))

            self.train_dataset = SquadDataset(**train_objects)
            self.eval_dataset = SquadDataset(**eval_objects)
        elif stage == 'test':
            test_objects = torch.load(
                self._gen_dataset_filename(self.test_split))

            self.test_dataset = SquadDataset(**test_objects)

    def post_process_data(self,
                          predicted_starts,
                          predicted_ends,
                          phase='eval'):
        post_processor = SquadBertAnsPostprocessor()
        dataset = self.eval_dataset if phase == 'eval' else self.test_dataset

        ans_predicted, ans_start_predicted, ans_end_predicted = post_processor(
            predicted_starts, predicted_ends, dataset.contexts_raw,
            dataset.input_features, dataset.tok2char)

        return dataset.answers, ans_predicted

    def _gen_dataset_filename(self, split: str):
        suffix = f'-{self.data_key}' if self.data_key else ''
        tokenizer_name = self.tokenizer_name.replace('/', '-')

        file_name = (f'{self.dataset_name}-{split}-{tokenizer_name}'
                     f'-{self.max_seq_length}{suffix}.ds')

        return os.path.join(self.data_dir, file_name)


if __name__ == '__main__':
    SquadDataModule('sbersquad', 'DeepPavlov/rubert-base-cased', '/tmp', 32,
                    384, 0).prepare_data()
