import argparse
import logging
import os

from crosslangt.nli.model import BERTNLIFineTuneModel
from transformers import BertForSequenceClassification
from tokenizers import BertWordPieceTokenizer

from pytorch_lightning import (
    seed_everything,
    Trainer,
)

from pytorch_lightning.callbacks import ModelCheckpoint


logger = logging.getLogger('finetune_mnli')


def finetune(hyperparameters):
    log_level = logging.WARN

    if hyperparameters.debug:
        log_level = logging.DEBUG
    elif hyperparameters.verbose:
        log_level = logging.INFO

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
    )

    # Make it reproducible.
    seed_everything(hyperparameters.seed)

    # Checkpoints
    checkpoint_file_fmt = os.path.join(
        hyperparameters.checkpoint_path,
        '{epoch}-{avg_val_acc:.4f}')

    model_checkpoint = ModelCheckpoint(checkpoint_file_fmt,
                                       save_weights_only=True,
                                       save_top_k=-1)

    model = BERTNLIFineTuneModel(hyperparameters)
    trainer = Trainer.from_argparse_args(hyperparameters,
                                         checkpoint_callback=model_checkpoint)

    if hyperparameters.train is True:
        logger.info('About to finte tune model.')
        trainer.fit(model)

    if hyperparameters.test is True:
        logger.info('About to test model.')
        trainer.test(model)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=54321)
    parser.add_argument('--checkpoint_path', type=str,
                        default='data/checkpoints')

    parser = BERTNLIFineTuneModel.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    params = parser.parse_args()

    finetune(params)


if __name__ == '__main__':
    main()
