import os
from typing import Tuple

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import Accuracy
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer)

from crosslangt.nli import (load_nli_dataset, prepare_nli_dataset)
from crosslangt.lexical import setup_lexical_for_training


class NLIModel(LightningModule):
    def __init__(self,
                 pretrained_model: str,
                 num_classes: int,
                 train_lexical_strategy: str,
                 train_dataset: str,
                 eval_dataset: str,
                 data_dir: str,
                 batch_size: int,
                 max_seq_length: int,
                 tokenizer_name: str = None,
                 **kwargs) -> None:

        super(NLIModel, self).__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(pretrained_model,
                                                 num_labels=num_classes)

        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model, config=self.config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name or self.hparams.pretrained_model)

        self.metric = Accuracy(num_classes=num_classes)

        self.__set_feature_keys()

    def forward(self, **inputs):
        return self.model(**inputs)

    def configure_optimizers(self):
        return Adam(self.bert.parameters(), lr=2e-5)

    def training_step(self, batch, batch_idx):
        loss, accuracy = self._run_step(batch, batch_idx)

        logs = {'train_loss': loss, 'train_acc': accuracy}
        tensor_bar = {'train_acc': accuracy}

        return {'loss': loss, 'log': logs, 'progress_bar': tensor_bar}

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._run_step(batch, batch_idx)

        logs = {'val_acc': accuracy, 'val_loss': loss}
        return {'val_loss': loss, 'val_acc': accuracy, 'log': logs,
                'progress_bar': logs}

    def validation_epoch_end(self, outputs):
        return self._eval_epoch_end(outputs, 'val_')

    def test_step(self, batch, batch_idx):
        loss, accuracy = self._run_step(batch, batch_idx)

        logs = {'test_acc': accuracy, 'test_loss': loss}
        return {'test_loss': loss, 'test_acc': accuracy, 'log': logs,
                'progress_bar': logs}

    def test_epoch_end(self, outputs):
        return self._eval_epoch_end(outputs, 'test_')

    def _run_step(self, batch, batch_idx):
        inputs = {}

        inputs['input_ids'] = batch['input_ids']
        inputs['attention_mask'] = batch['attention_mask']
        inputs['labels'] = batch['label']

        if self.config.model_type in ['bert', 'xlnet', 'albert']:
            inputs['token_type_ids'] = batch['token_type_ids']

        outputs = self(**inputs)

        loss, logits = outputs[:2]
        predicted = torch.argmax(logits, dim=-1)

        accuracy = self.metric(predicted, batch['label'])

        return loss, accuracy

    def _eval_epoch_end(self, outputs, prefix):
        acc_key = f'{prefix}acc'
        loss_key = f'{prefix}loss'

        accuracies = torch.stack([o[acc_key] for o in outputs])
        losses = torch.stack([o[loss_key] for o in outputs])

        mean_accuracy = accuracies.mean()
        mean_loss = losses.mean()

        return {f'{prefix}avg_accuracy': mean_accuracy,
                f'{prefix}avg_loss': mean_loss}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          self.hparams.batch_size,
                          shuffle=True,
                          num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.eval_dataset,
                          self.hparams.batch_size,
                          shuffle=False,
                          num_workers=8)

    def prepare_data(self) -> None:
        prepare_nli_dataset(dataset=self.hparams.train_dataset,
                            split='train',
                            data_dir=self.hparams.data_dir,
                            tokenizer=self.tokenizer,
                            max_seq_length=self.hparams.max_seq_length,
                            features_key=self.train_key,
                            force=True)

        prepare_nli_dataset(dataset=self.hparams.eval_dataset,
                            split='eval',
                            data_dir=self.hparams.data_dir,
                            tokenizer=self.tokenizer,
                            max_seq_length=self.hparams.max_seq_length,
                            features_key=self.train_key,
                            force=True)

    def setup(self, stage: str):
        setup_lexical_for_training(self.hparams.train_lexical_strategy,
                                   self.bert, self.tokenizer)

        self.train_dataset = load_nli_dataset(
            self.hparams.data_dir,
            self.hparams.train_dataset,
            'train',
            self.hparams.max_seq_length,
            self.train_key)

        self.eval_dataset = load_nli_dataset(
            self.hparams.data_dir,
            self.hparams.eval_dataset,
            'eval',
            self.hparams.max_seq_length,
            self.train_key)

    def __set_feature_keys(self, key_type):
        model_name = self.hparams.pretrained_model.split('/').pop()
        tokenizer_name = self.hparams.tokenizer_name or ''
        tokenizer_name = tokenizer_name.split('/').pop()

        self.train_key = f'{model_name}.{tokenizer_name}'
