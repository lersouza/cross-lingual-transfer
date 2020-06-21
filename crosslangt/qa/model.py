import os
import torch

from crosslangt.basemodel import BERTFineTuneModel
from torch.utils.data import DataLoader
from transformers import BertForQuestionAnswering

from transformers.data import (
    squad_convert_examples_to_features,
    SquadV1Processor )


class BERTSQuADFinetuneModel(BERTFineTuneModel):
    def __init__(self, hparams):
        super().__init__(hparams)

    def _build_bert_model(self):
        self.bert_model = BertForQuestionAnswering.from_pretrained(
            self.hparams.model)

    def forward(self, input_ids, attention_mask, token_type_ids,
                start_positions=None, end_positions=None):

        outputs = self.bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions)

        return outputs

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, \
            start_positions, end_positions = batch

        outputs = self(input_ids, attention_mask, token_type_ids,
                       start_positions, end_positions)

        loss = outputs[0]
        logs = {'train_loss': loss}

        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, \
            start_positions, end_positions = batch

        loss, start_logits, end_logits = self(
            input_ids,
            attention_mask,
            token_type_ids,
            start_positions,
            end_positions)

        logs = {'val_loss': loss}
        
        return {'val_loss': loss, 'log': logs,
                'start_logits': start_logits, 'end_logits': end_logits}

    def validation_epoch_end(self, outputs):
        start_logits = torch.stack([x['start_logits'] for x in outputs])
        end_logits = torch.stack([x['end_logits'] for x in outputs])

        return {}

    def test_step(self, batch, batch_idx):
        pass

    def test_epoch_end(self, outputs):
        pass

    def train_dataloader(self):
        return self.__get_dataloader('train', True)

    def val_dataloader(self):
        return self.__get_dataloader('eval')

    def test_dataloader(self):
        return self.__get_dataloader('test')

    def __build_dataset(self, dataset_type: str):
        processor = SquadV1Processor()
        data_dir = self.hparams.data_dir
        examples = None

        if dataset_type == 'train':
            examples = processor.get_train_examples(
                data_dir, self.hparams.train_file)
        elif dataset_type == 'eval':
            examples = processor.get_dev_examples(
                data_dir, self.hparams.eval_file)
        elif dataset_type == 'test':
            examples = processor.get_test_examples(
                data_dir)

        if examples is None:
            raise ValueError(f'Unrecognized dataset {dataset_type}')

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=self.tokenizer,
            max_seq_length=self.hparams.max_seq_length,
            doc_stride=self.hparams.doc_stride,
            max_query_length=self.hparams.max_query_length,
            is_training=True,
            return_dataset='pt'
        )

        return (features, dataset, examples)

    def __get_dataloader(self, dataset_type: str, shuffle: bool = False):
        model_ind = self.hparams.model.split('/')[-1]

        cached_path = os.path.join(
            self.hparams.dataset_cache_path,
            f'cached_{dataset_type}_{model_ind}_{self.hparams.max_seq_length}')

        if os.path.exists(cached_path):
            features_and_dataset = torch.load(cached_path)
            dataset = features_and_dataset['dataset']
        else:
            features, dataset, examples = self.__build_dataset(dataset_type)
            torch.save({
                'features': features,
                'dataset': dataset,
                'examples': examples}, cached_path)
        
        return DataLoader(
           dataset, shuffle=shuffle, batch_size=self.hparams.batch_size)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = BERTFineTuneModel.add_model_specific_args(parent_parser)

        parser.add_argument('--dataset_cache_path', type=str,
                            default='data/.cache/squad/',
                            help='Location for caching the processed dataset')

        parser.add_argument('--data_dir', type=str,
                            default='data/squad/',
                            help='Directory where squad data is located.')

        parser.add_argument('--train_file', type=str,
                            default=None, help='Train file name.')

        parser.add_argument('--eval_file', type=str, default=None,
                            help='Eval file name.')

        parser.add_argument('--dataset_build_threads', type=int, default=2,
                            help='No. of threads used to process dataset.')

        parser.add_argument('--max_seq_length', type=int, default=384,
                            help='Maximum length of input sequence.')

        parser.add_argument('--doc_stride', type=int, default=128,
                            help='Doc stride for generating sub-parts of '
                                 'long squad contexts.')

        parser.add_argument('--max_query_length', type=int, default=64,
                            help='Max length for the query input part.')

        return parser
