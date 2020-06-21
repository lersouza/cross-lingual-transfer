import logging
import os
import torch

from argparse import ArgumentParser
from crosslangt import tokenization, basemodel
from crosslangt.metrics import compute_accuracy
from torch.utils.data import DataLoader
from transformers.data.data_collator import DefaultDataCollator

from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast
)
from transformers.data.datasets import (
    GlueDataset,
    GlueDataTrainingArguments,
)


class BERTNLIFineTuneModel(basemodel.BERTFineTuneModel):
    """
    A model for finetuning BERT in NLI tasks.
    """
    def __init__(self, hparams):
        super().__init__(hparams)

    def _build_bert_model(self):
        bert_config = BertConfig.from_pretrained(
            self.hparams.model,
            num_labels=3,
            output_hidden_states=False,
            output_attentions=False,
            finetuning_task='mnli')

        self.bert_model = BertForSequenceClassification.from_pretrained(
            self.hparams.model, config=bert_config)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        """
        Run the model.
        """
        shapes = (input_ids.shape, attention_mask.shape, token_type_ids.shape)
        self.log.debug(f'BERT-MNLI: input shapes are: {shapes}')

        bert_output = self.bert_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels)

        return bert_output

    def training_step(self, batch, batch_idx):
        """ Runs a training step for the specified (mini-)batch. """
        outputs = self(**batch)

        # When labels are provided, the first item in the output tuple
        # is the calculated loss for the batch
        loss = outputs[0]
        logs = {'training_loss': loss}

        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        """ Runs a validation step for this mode. """
        loss, logits = self(**batch)

        val_acc = compute_accuracy(logits, batch['labels'])
        logs = {'val_loss': loss, 'val_acc': val_acc}

        return {'val_loss': loss, 'val_acc': val_acc, 'log': logs}

    def validation_epoch_end(self, outputs):
        """ Aggregates metrics for validation epoch. """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        bar = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}
        return {
            'val_loss': avg_loss,
            'avg_val_acc': avg_val_acc,
            'progress_bar': bar}

    def test_step(self, batch, batch_idx):
        """ Executes a test step in the model. """
        model_output = self(**batch)

        if 'labels' in batch:
            _, logits = model_output
            test_acc = compute_accuracy(logits, batch['labels'])
            logs = {'test_acc': test_acc}

            return {'test_acc': test_acc, 'log': logs}
        else:
            logits = model_output[0]

            # For MNLI dataset, labels are not available
            # in the test dataset. So, we need to generate a file
            # for submiting to Open Kaggle Competition.
            predictions = torch.argmax(logits, dim=-1)
            return {'test_predictions': predictions}

    def test_epoch_end(self, outputs):
        """ Consolidate all outputs from mini-batches. """
        if 'test_predictions' in outputs:
            all_predictions = torch.stack(
                [x['test_predictions'] for x in outputs])

            output_file = os.path.join(
                self.hparams.predicted_output_dir,
                f'test-predictions-{self.current_epoch}.csv')

            with open(output_file, 'w+') as file:
                for i, prediction in all_predictions:
                    file.write(i)
                    file.write(',')
                    file.write(
                        self.test_dataset.get_labels()[prediction.detach()])
                    file.write('\n')

            return {'total_predictions': len(all_predictions)}
        elif 'test_acc' in outputs:
            avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
            return {'avg_test_acc': avg_test_acc}

        return {}  # Just in case test has produced no outputs

    def train_dataloader(self):
        """ Returns a DataLoader for training the Model. """
        dataset = self.train_dataset or self._build_dataset('train')
        return self._build_dataloader(dataset, True)

    def val_dataloader(self):
        """ Returns a DataLoader for validating the Model. """
        dataset = self.eval_dataset or self._build_dataset('dev')
        return self._build_dataloader(dataset)

    def test_dataloader(self):
        """ Returns a DataLoader for testing the Model. """
        dataset = self.test_dataset or self._build_dataset('test')
        return self._build_dataloader(dataset)

    def _build_dataset(self, dataset_type):
        """ Builds a GlueDataset for the specified `dataset_type`. """
        data_args = GlueDataTrainingArguments(
            task_name='mnli',
            max_seq_length=self.hparams.max_seq_length,
            data_dir=self.hparams.data_dir)

        dataset = GlueDataset(
            data_args, self.tokenizer, self.hparams.limit_data, dataset_type)

        return dataset

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = basemodel.BERTFineTuneModel.add_model_specific_args(
            parent_parser)

        parser.add_argument('--data_dir', metavar='PATH',
                            type=str, default='data/mnli',
                            help='The dir where dataset files are located.')

        parser.add_argument('--max_seq_length', metavar='S',
                            type=int, default=128,
                            help='Max seq length for inputs to the model.')

        parser.add_argument('--limit_data', metavar='N',
                            type=int, default=None,
                            help='The maximum examples to use from dataset.')

        return parser
