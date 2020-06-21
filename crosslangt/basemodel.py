import logging
import torch

from argparse import ArgumentParser
from crosslangt.tokenization import get_tokenizer
from pytorch_lightning import LightningModule
from torch.nn import Embedding
from torch.utils.data import DataLoader
from transformers import BertPreTrainedModel, BertConfig
from transformers.data.data_collator import DefaultDataCollator


class BERTFineTuneModel(LightningModule):
    """ Base class for fine tuning models, with some utility methods. """
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.log = logging.getLogger(self.__class__.__name__)

        self._build_bert_model()

        if hparams.lexical:
            self.load_pretrained_lexical()

        if hparams.freeze_lexical:
            self.freeze_lexical()

        self.tokenizer = get_tokenizer(
            hparams.model, hparams.vocab, type='transformers')

        self._init_datasets(None, None, None)

    def _init_datasets(self, train_dataset, eval_dataset, test_dataset):
        """ Sets the class datasets to be used. """
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset

    def _build_bert_model(self):
        """
        Abstract method for child classes to build its BERT Model.
        The model should be set into `self.bert_model` attribute.
        """
        pass

    def _build_dataloader(self, dataset, shuffle: bool = False,
                          collate_fn=None):
        """
        Builds a dataloader for given `dataset` using experiment hyper params.
        """

        if collate_fn is None:
            collator = DefaultDataCollator()
            collate_fn = collator.collate_batch

        loader = DataLoader(dataset, shuffle=shuffle,
                            batch_size=self.hparams.batch_size,
                            collate_fn=collate_fn,
                            num_workers=self.hparams.dataloader_workers)

        return loader

    def freeze_lexical(self):
        """ Freezes the lexical part of Bert Model. """
        self.log.info('BERT-MNLI: Freezing BERT model lexical. '
                      'All Input Embeddings will not be updated.')

        embeddings = self.bert_model.get_input_embeddings()

        for parameter in embeddings.parameters():
            parameter.requires_grad = False

    def load_pretrained_lexical(self):
        """ Loads a pre-trained lexical layer specified in hyperparams. """
        lexical_state = torch.load(self.hparams.lexical)
        weights = lexical_state['weight']

        embeddings = Embedding(weights.shape[0], weights.shape[1],
                               padding_idx=self.tokenizer.pad_token_id)
        embeddings.load_state_dict(weights)

        self.bert_model.set_input_embeddings(embeddings)

    def configure_optimizers(self):
        """
        Configure the optimizer.
        For this experiment, we always use Adam without decay.
        """
        lr = self.hparams.lr

        self.log.info(f'Using Adam Optimizer with learning rate={lr}.')

        return torch.optim.Adam(self.parameters(), lr=lr)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument('--model', type=str, default='bert-base-cased',
                            help='The model name or path to fine tune.')

        parser.add_argument('--vocab', type=str, default=None,
                            help='The path to a vocabulary to use. If not '
                                 'specified, model\'s pre trained '
                                 'will be used.')

        parser.add_argument('--lexical', type=str, default=None,
                            help='Path for a pre-trained lexical use.')

        parser.add_argument('--freeze_lexical', action='store_true',
                            help='Freezes the model\'s Lexical part.')

        parser.add_argument('--lr', type=float, default=3e-5,
                            help='The Learning rate to fine tune.')

        parser.add_argument('--batch_size', metavar='B',
                            type=int, default=32,
                            help='The batch size for loading.')

        parser.add_argument('--dataloader_workers', type=int, default=8,
                            help='Number of workers for data loading.')

        return parser
