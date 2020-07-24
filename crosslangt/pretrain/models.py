import os

import torch
from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.serialization import validate_cuda_device
from torch.utils.data.dataloader import DataLoader
from transformers import BertForPreTraining, BertTokenizer

from crosslangt.pretrain.dataset import LexicalTrainDataset


class LexicalTrainingModel(LightningModule):
    def __init__(self,
                 pretrained_model,
                 data_dir: str,
                 batch_size: int) -> None:
        super(LexicalTrainingModel, self).__init__()

        self.save_hyperparameters()

        self.bert = BertForPreTraining.from_pretrained(pretrained_model)
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    def forward(self, input_ids, attention_mask, token_type_ids,
                masked_lm_labels=None, next_sentence_label=None):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            masked_lm_labels=masked_lm_labels,
            next_sentence_label=next_sentence_label
        )

        return outputs

    def configure_optimizers(self):
        return Adam(self.parameters(),
                    lr=1e-4,
                    betas=(0.9, 0.999),
                    eps=1e-6,
                    weight_decay=0.01)

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

        outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            masked_lm_labels=masked_lm_labels,
            next_sentence_label=next_sentence_label
        )
        loss = outputs[0]

        return loss

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=self.train_dataset.collate_batch)

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=self.eval_dataset.collate_batch)

    def setup(self, stage: str):
        if stage == 'fit':
            tindex = os.path.join(self.hparams.data_dir, 'train_index')
            eindex = os.path.join(self.hparams.data_dir, 'eval_index')

            self.train_dataset = LexicalTrainDataset(tindex, self.tokenizer)
            self.eval_dataset = LexicalTrainDataset(eindex, self.tokenizer)
        else:
            tindex = os.path.join(self.hparams.data_dir, 'test_index')
            self.test_dataset = LexicalTrainDataset(tindex, self.tokenizer)