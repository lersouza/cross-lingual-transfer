import argparse

from model import BERTNLIModel
from dataset import (
    NLIDataset, load_mnli_dataset, load_assin_dataset
)


def fine_tune(hyperparams):
    model = BERTNLIModel(hyperparams)


def main():
    parser = argparse.ArgumentParser()

    # Model name or path
    parser.add_argument(
        '--base_model_name',
        default='bert-base-cased',
        type=str,
        help='The pre-trained model name for fine-tuning')

    parser.add_argument(
        '--data-path',
        default='./',
        type=str,
        help='The base path where dataset files are located.')

    parser.add_argument(
        '--train-data-file',
        default='train_matched.tsv',
        type=str,
        help='The train dataset file name.')

    parser.add_argument(
        '--val-data-file',
        default='dev_matched.tsv',
        type=str,
        help='The validation dataset file name.')



    args = parser.parse_args()
    fine_tune(args)


if __name__ == "__main__":
    main()
