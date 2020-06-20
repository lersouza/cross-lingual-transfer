import logging
import torch
import pytorch_lightning as pl

from argparse import Namespace
from crosslangt import tokenization
from sklearn.metrics import accuracy_score
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


class BERTNLIFineTuneModel(pl.LightningModule):
    """
    A model for finetuning BERT in NLI tasks.
    """
    def __init__(self, hparams: Namespace):
        super().__init__()

        self.hparams = hparams

        # Modules
        bert_config = BertConfig.from_pretrained(
            hparams.model,
            num_labels=hparams.num_labels,
            output_hidden_states=False,
            output_attentions=False,
            finetuning_task='mnli')

        self.bert_model = BertForSequenceClassification.from_pretrained(
            hparams.model, config=bert_config)

        if hparams.freeze_lexical:
            self.freeze_lexical()

        self.tokenizer = tokenization.get_tokenizer(
            self.hparams.model, self.hparams.vocab, type='transformers')

        # Datasets
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

    def freeze_lexical(self):
        """ Freezes the lexical part of Bert Model. """
        logger.info('BERT-MNLI: Freezing BERT model lexical. '
                    'All Input Embeddings will not be updated.')

        embeddings = self.bert_model.get_input_embeddings()

        for parameter in embeddings.parameters():
            parameter.requires_grad = False

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        """
        Run the model.
        """
        shapes = (input_ids.shape, attention_mask.shape,
                  token_type_ids.shape, labels.shape)

        logger.debug(f'BERT-MNLI: input shapes are: {shapes}')

        bert_output = self.bert_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels)

        return bert_output

    def configure_optimizers(self):
        """
        Configure the optimizer.
        For this experiment, we always use Adam without decay.
        """
        lr = self.hparams.lr
        params = list(self.parameters())

        logger.info(f'Using Adam Optimizer for {len(params)} parameters '
                    f'with learning rate={lr}.')

        return torch.optim.Adam(params, lr=lr)

    def calculate_accuracy(self, logits, y):
        """
        Calculates accuracy obtained from logits,
        based on actual `y` labels.
        """
        # Logits shape should be (B, C), where B is Batch Size
        # and C is the number of classes the model outputs.
        predicted = torch.argmax(logits, dim=1)
        accuracy = accuracy_score(predicted.cpu(), y.cpu())

        return torch.tensor(accuracy)

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

        val_acc = self.calculate_accuracy(logits, batch['labels'])
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

        if self.hparams.do_test:
            self.test_dataset = GlueDataset(
                training_args, self.tokenizer, limit_data_length, 'test')

    def train_dataloader(self):
        return self.__build_dataloader(self.train_dataset, True)

    def val_dataloader(self):
        return self.__build_dataloader(self.eval_dataset)

    def __build_dataloader(self, dataset: GlueDataset, shuffle: bool = False):
        """
        Builds a dataloader for given `dataset` using experiment hyper params.
        """
        collator = DefaultDataCollator()

        loader = DataLoader(dataset, shuffle=shuffle,
                            batch_size=self.hparams.batch_size,
                            collate_fn=collator.collate_batch)

        return loader
