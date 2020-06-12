import argparse
import os

from pathlib import Path
from tokenizers import BertWordPieceTokenizer


def train_bert_tokenizer(root_source_path, target_path, tokenizer_name,
                         vocab_size=30000, lower_case=False):
    """
    Trains a BERT WordPiece Tokenizer based on data
    located under `root_path`.

    This function adds special tokens for compability with pre-trained
    tokenizers available in Transformers' library from Huggingface:
    - [PAD] in position 0
    - [unusedXXX] from position 1 to 99
    - [UNK], [CLS], [SEP] and [MASK] from position 100 to 103
    - [unused100] and [unused101], so vocab will start at 102
    """
    files = [str(f) for f in Path(root_source_path).glob('**/*')]
    tokenizer = BertWordPieceTokenizer(lowercase=lower_case,
                                       strip_accents=False,
                                       handle_chinese_chars=True)

    # We add some initial special tokens
    # this is intended to make our new tokenizer compatible
    # with the Pre-trained ones from Huggingface Transformers'
    # library.
    initial = ['[PAD]'] + \
              [f'[unused{i}]' for i in range(1, 100)] + \
              ['[UNK]', '[CLS]', '[SEP]', '[MASK]'] + \
              ['[unused100]', '[unused101]']

    tokenizer.train(files=files, special_tokens=initial,
                    vocab_size=vocab_size)

    tokenizer.save(target_path, tokenizer_name)


def main():
    """ Executes tokenizer training from sys.args. """
    parser = argparse.ArgumentParser()

    parser.add_argument('root_source_path', type=str,
                        help='The root directory where dataset files'
                             'are located.')

    parser.add_argument('target_path', type=str,
                        help='the directory where tokenizer '
                             'will be saved.')

    parser.add_argument('tokenizer_name', type=str,
                        help='the name of the tokenizer.')

    parser.add_argument('--vocab_size', type=int, default=30000,
                        help='the vocab size to generate (default=30k).')

    parser.add_argument('--lower_case', action='store_true',
                        help='indicates whether to perform lowercase on'
                             'vocabulary words.')

    args = parser.parse_args()
    train_bert_tokenizer(**vars(args))


if __name__ == '__main__':
    main()
