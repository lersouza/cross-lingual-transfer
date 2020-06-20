import argparse
import logging

from crosslangt.nli.model import BERTNLIFineTuneModel
from transformers import BertForSequenceClassification
from tokenizers import BertWordPieceTokenizer

from pytorch_lightning import (
    seed_everything,
    Trainer)


def finetune(hyperparameters):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    seed_everything(hyperparameters.seed)

    model = BERTNLIFineTuneModel(hyperparameters)
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(model)


def main():
    parser = argparse.ArgumentParser('Fine tune BERT to MNLI Dataset')

    parser.add_argument('--model', type=str, default='bert-base-cased',
                        help='The name of the model to finetune. '
                             'Default=bert-base-cased')

    parser.add_argument('--vocab', type=str, default=None,
                        help='The vocab file to use. Default=vocab.txt')

    parser.add_argument('--freeze_lexical', action='store_true',
                        help='Indicate whether to freeze the model lexical.')

    parser.add_argument('--num_labels', type=int, default=3,
                        help='Number of classes to output. Default = 3')

    parser.add_argument('--data_dir', type=str, default='data/mnli/',
                        help='The directory where dataset files are located.')

    parser.add_argument('--do_test', action='store_true',
                        help='Indicate whether a test should be performed.')

    parser.add_argument('--seed', type=int, default=54321,
                        help='The random seed for the experiment.')

    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for running the model, per gpu.')

    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate to use in fine tuning.')

    params = parser.parse_args()

    finetune(params)


if __name__ == '__main__':
    main()
