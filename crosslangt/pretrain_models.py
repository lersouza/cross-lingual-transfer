import torch
import pytorch_lightning as pl

from argparse import Namespace
from transformers import BertConfig, BertModel


class BertForMaskedLM(pl.LightningModule):
    """
    This class defines the BERT Model with and pre-training procedures
    for this experiment.
    """

    def __init__(self, hparams: Namespace):
        """ Initialize the model with specified hyperparameters (hparams) """
        pass

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        pass

    def configure_optimizers(self):
        pass
