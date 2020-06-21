import logging
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


logger = logging.getLogger(__name__)


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
        shapes = (input_ids.shape, attention_mask.shape,
                  token_type_ids.shape, labels.shape)

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
        return {'val_loss': avg_loss, 'progress_bar': bar}

    def prepare_data(self):
        """ Load the datasets to be used. """
        training_args = GlueDataTrainingArguments(
            task_name='mnli',
            data_dir=self.hparams.data_dir)

        limit_data_length = getattr(self.hparams, 'limit_data', None)

        self.train_dataset = GlueDataset(
            training_args, self.tokenizer, limit_data_length, 'train')

        self.eval_dataset = GlueDataset(
            training_args, self.tokenizer, limit_data_length, 'dev')

        self.test_dataset = GlueDataset(
            training_args, self.tokenizer, limit_data_length, 'test')

    def train_dataloader(self):
        return self._build_dataloader(self.train_dataset, True)

    def val_dataloader(self):
        return self._build_dataloader(self.eval_dataset)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = basemodel.BERTFineTuneModel.add_model_specific_args(
            parent_parser)

        parser.add_argument('--data_dir', metavar='PATH',
                            type=str, default='data/mnli',
                            help='The dir where dataset files are located.')

        parser.add_argument('--limit_data', metavar='N',
                            type=int, default=None,
                            help='The maximum examples to use from dataset.')

        return parser
