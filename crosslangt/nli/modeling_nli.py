import os

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import Accuracy
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from transformers import (BertConfig, BertForSequenceClassification,
                          BertTokenizer)

from crosslangt.nli import (load_nli_dataset, prepare_nli_dataset)
from crosslangt.lexical import (setup_lexical_for_testing,
                                setup_lexical_for_training)


class NLIModel(LightningModule):
    def __init__(self,
                 pretrained_model: str,
                 num_classes: int,
                 train_lexical_strategy: str,
                 test_lexical_strategy: str,
                 train_dataset: str,
                 test_dataset: str,
                 data_dir: str,
                 batch_size: int,
                 max_seq_length: int,
                 test_lexical_path: str = None,
                 tokenizer_name: str = None,
                 test_tokenizer_name: str = None,
                 **kwargs) -> None:

        super(NLIModel, self).__init__()

        self.save_hyperparameters()

        config = BertConfig.from_pretrained(pretrained_model,
                                            num_labels=num_classes)

        self.bert = BertForSequenceClassification.from_pretrained(
            pretrained_model, config=config)

        self.train_tokenizer, self.test_tokenizer = self.__get_tokenizers()
        self.metric = Accuracy(num_classes=num_classes)

        self.training_setup_performed = False
        self.test_setup_performed = False

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            labels=labels)

        return outputs

    def configure_optimizers(self):
        return Adam(self.bert.parameters(), lr=2e-5)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = \
            batch['input_ids'], batch['attention_mask'], \
            batch['token_type_ids'], batch['label']

        outputs = self(input_ids, attention_mask, token_type_ids, labels)
        loss = outputs[0]
        logits = outputs[1]
        predicted = torch.argmax(logits, dim=-1)

        accuracy = self.metric(predicted, labels)

        logs = {'train_loss': loss, 'train_acc': accuracy}
        tensor_bar = {'train_acc': accuracy}

        return {'loss': loss, 'log': logs, 'progress_bar': tensor_bar}

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = \
            batch['input_ids'], batch['attention_mask'], \
            batch['token_type_ids'], batch['label']

        outputs = self(input_ids, attention_mask, token_type_ids)
        logits = outputs[0]
        predicted = torch.argmax(logits, dim=-1)

        accuracy = self.metric(predicted, labels)

        logs = {'test_acc': accuracy}
        return {'test_acc': accuracy, 'log': logs, 'progress_bar': logs}

    def test_epoch_end(self, outputs):
        accuracies = torch.stack([o['test_acc'] for o in outputs])
        mean_accuracy = accuracies.mean()

        return {'test_avg_accuracy': mean_accuracy}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          self.hparams.batch_size,
                          shuffle=True,
                          num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          self.hparams.batch_size,
                          shuffle=False,
                          num_workers=8)

    def prepare_data(self) -> None:
        prepare_nli_dataset(dataset=self.hparams.train_dataset,
                            split='train',
                            data_dir=self.hparams.data_dir,
                            tokenizer=self.train_tokenizer,
                            max_seq_length=self.hparams.max_seq_length,
                            features_key=self.hparams.pretrained_model,
                            force=True)

        prepare_nli_dataset(dataset=self.hparams.test_dataset,
                            split='eval',
                            data_dir=self.hparams.data_dir,
                            tokenizer=self.test_tokenizer,
                            max_seq_length=self.hparams.max_seq_length,
                            features_key=self.hparams.pretrained_model,
                            force=True)

    def setup(self, stage: str):
        if stage == 'fit' and not self.training_setup_performed:
            setup_lexical_for_training(self.hparams.train_lexical_strategy,
                                       self.bert, self.train_tokenizer)

            self.train_dataset = load_nli_dataset(self.hparams.train_dataset,
                                                  'train',
                                                  self.hparams.data_dir,
                                                  self.hparams.max_seq_length)

            self.training_setup_performed = True
        elif stage == 'test' and not self.test_setup_performed:
            setup_lexical_for_testing(self.hparams.test_lexical_strategy,
                                      self.bert, self.test_tokenizer,
                                      self.hparams.test_lexical_path)

            self.test_dataset = load_nli_dataset(self.hparams.test_dataset,
                                                 'eval', self.hparams.data_dir,
                                                 self.hparams.max_seq_length)

            self.test_setup_performed = True

    def __get_tokenizers(self):
        train_tokenizer = BertTokenizer.from_pretrained(
            self.hparams.tokenizer_name or self.hparams.pretrained_model)

        test_tokenizer = train_tokenizer  # By default, they are the same.

        if self.hparams.test_tokenizer_name is not None:
            test_tokenizer = BertTokenizer.from_pretrained(
                self.hparams.test_tokenizer_name)

        return (train_tokenizer, test_tokenizer)
