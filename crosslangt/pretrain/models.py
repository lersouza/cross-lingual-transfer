from crosslangt.lexical import SlicedEmbedding, SlicedOutputEmbedding
import logging
import os
import torch

from pytorch_lightning import LightningModule
from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from transformers import BertConfig, BertForPreTraining, BertTokenizer

from crosslangt.dataset_utils import download_and_extract
from crosslangt.pretrain.dataset import LexicalTrainDataset


logger = logging.getLogger(__name__)


class LexicalTrainingModel(LightningModule):
    google_checkpoint_location = \
        'https://storage.googleapis.com/bert_models/2018_10_18/' \
        'cased_L-12_H-768_A-12.zip'

    google_checkpoint_root = 'cased_L-12_H-768_A-12'

    def __init__(self,
                 pretrained_model,
                 tokenizer_name_or_path: str,
                 data_dir: str,
                 batch_size: int,
                 max_train_examples: int = None,
                 max_eval_examples: int = None,
                 train_strategy='train-all-lexical') -> None:
        super(LexicalTrainingModel, self).__init__()

        self.save_hyperparameters()

        if pretrained_model.startswith('google-checkpoint'):
            self._load_google_checkpoint()
        else:
            self.bert = BertForPreTraining.from_pretrained(pretrained_model)

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name_or_path)

        self.__setup_lexical_for_training()

        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None

    def forward(self,
                input_ids,
                attention_mask,
                token_type_ids,
                masked_lm_labels=None,
                next_sentence_label=None):

        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            masked_lm_labels=masked_lm_labels,
                            next_sentence_label=next_sentence_label)

        return outputs

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)

    def training_step(self, batch, batch_idx):
        loss = self.__run_step(batch)
        logs = {'train_loss': loss}

        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        loss = self.__run_step(batch)
        perplexity = torch.exp(loss)

        logs = {'val_loss': loss, 'val_perplexity': perplexity}
        return {'val_loss': loss, 'val_perplexity': perplexity, 'log': logs}

    def validation_epoch_end(self, outputs):
        all_perplexity = torch.stack([o['val_perplexity'] for o in outputs])
        avg_perplexity = all_perplexity.mean()

        return {'avg_val_perplexity': avg_perplexity}

    def __run_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        token_type_ids = batch['token_type_ids']
        masked_lm_labels = batch['mlm_labels']
        next_sentence_label = batch['next_sentence_label']

        outputs = self(input_ids=input_ids,
                       attention_mask=attention_mask,
                       token_type_ids=token_type_ids,
                       masked_lm_labels=masked_lm_labels,
                       next_sentence_label=next_sentence_label)
        loss = outputs[0]

        return loss

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=8,
                          shuffle=True,
                          collate_fn=self.train_dataset.collate_batch)

    def val_dataloader(self):
        return DataLoader(self.eval_dataset,
                          batch_size=self.hparams.batch_size,
                          num_workers=8,
                          shuffle=False,
                          collate_fn=self.eval_dataset.collate_batch)

    def setup(self, stage: str):
        if stage == 'fit' and self.train_dataset is None:
            tindex = os.path.join(self.hparams.data_dir, 'train_index')
            eindex = os.path.join(self.hparams.data_dir, 'eval_index')

            self.train_dataset = LexicalTrainDataset(
                tindex,
                self.tokenizer,
                max_examples=self.hparams.max_train_examples)

            self.eval_dataset = LexicalTrainDataset(
                eindex,
                self.tokenizer,
                max_examples=self.hparams.max_eval_examples)

            self.__setup_lexical_for_training()
        else:
            tindex = os.path.join(self.hparams.data_dir, 'test_index')
            self.test_dataset = LexicalTrainDataset(tindex, self.tokenizer)

    def _load_google_checkpoint(self):
        logger.info('Loading Checkpoint from Google for Pre training')

        download_and_extract(self.google_checkpoint_location, './')

        checkpoint_dir = os.path.join('./', self.google_checkpoint_root)
        config_location = os.path.join(checkpoint_dir, 'bert_config.json')
        index_location = os.path.join(checkpoint_dir, 'bert_model.ckpt.index')

        logger.info(
            f'Config file: {config_location}. Index file: {index_location}')

        config = BertConfig.from_json_file(config_location)
        self.bert = BertForPreTraining.from_pretrained(index_location,
                                                       config=config,
                                                       from_tf=True)

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def __setup_lexical_for_training(self):
        if self.hparams.train_strategy == 'train-all-lexical':
            # We freeze all parameters in this model.
            # Then, we unlock the ones we want.
            self._freeze_parameters()

            # Train Word Embeddings Only
            input_embeddings = self.bert.get_input_embeddings()

            for parameter in input_embeddings.parameters():
                parameter.requires_grad = True

            # We also train the HEAD (Output Embeddings, since they are tied)
            output_embeddings = self.bert.get_output_embeddings()

            for parameter in output_embeddings.parameters():
                parameter.requires_grad = True

        elif self.hparams.train_strategy == 'train-non-special':
            self._freeze_parameters()

            # Train Word Embeddings Only, skipping special tokens
            input_embeddings = self.bert.get_input_embeddings()
            last_special_token = max(self.tokenizer.all_special_ids)

            new_input_embeddings = SlicedEmbedding.slice(
                input_embeddings, last_special_token + 1, True, False)

            self.bert.set_input_embeddings(new_input_embeddings)

            # Handling output embeddings
            output_embeddings = self.bert.get_output_embeddings()

            new_output_embeddings = SlicedOutputEmbedding(
                output_embeddings, last_special_token + 1, True, False)

            self.bert.cls.predictions.decoder = new_output_embeddings
