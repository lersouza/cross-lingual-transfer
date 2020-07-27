import json
import os

from typing import List
from tokenizers import BertWordPieceTokenizer
from transformers.tokenization_utils import TOKENIZER_CONFIG_FILE


def get_bert_initial_alphabet():
    """
    Returns the initial tokens of BERT vocabulary.
    BERT vocab has some unused tokens and specific position for
    special ones. So we return them in order, so it makes our tokenizer
    compatible with pre-trained BERT Tokenizers.
    """
    initial_tokens =  \
        ['[PAD]'] + \
        [f'[unused{i}]' for i in range(1, 100)] + \
        ['[UNK]', '[CLS]', '[SEP]', '[MASK]'] + \
        ['[unused100]', '[unused101]']

    return initial_tokens


def train_tokenizer(files: List[str],
                    tokenizer_name: str,
                    base_path: str,
                    vocab_size: int,
                    lowercase: bool = False,
                    strip_accents: bool = False):

    tokenizer = BertWordPieceTokenizer(
        lowercase=lowercase,
        strip_accents=strip_accents)

    tokenizer_path = os.path.join(base_path, tokenizer_name)
    os.makedirs(tokenizer_path, exist_ok=True)

    initial_alphabet = get_bert_initial_alphabet()

    tokenizer.train(files, initial_alphabet=initial_alphabet,
                    vocab_size=vocab_size)

    tokenizer.save(tokenizer_path)

    # Creating a default config for the tokenizer
    config = {'do_lower_case': lowercase}
    config_file_path = os.path.join(tokenizer_path, TOKENIZER_CONFIG_FILE)

    with open(config_file_path, 'w+') as config_file:
        json.dump(config, config_file)
