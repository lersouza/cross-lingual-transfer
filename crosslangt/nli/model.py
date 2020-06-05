import torch
import pytorch_lightning as pl

from argparse import Namespace
from transformers import (
    BertForSequenceClassification,
    BertTokenizer
)


class BERTNLIModel(pl.LightningModule):
    """
    A model for training and testing BERT in NLI tasks.
    """
    
    def __init__(self, hparams: Namespace):
        super().__init__()

        self.hparams = hparams

        # Modules
        self.bert_model = BertForSequenceClassification.from_pretrained(
            hparams.base_model_name)

        self.tokenizer = BertTokenizer.from_pretrained(hparams.base_model_name)


    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        """
        Run the model.
        """
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
        parameters = [p for p in self.parameters() if p.requires_grad]

        return torch.optim.Adam(parameters, lr=self.hparams.lr)
 
    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass




