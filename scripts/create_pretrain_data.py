import argparse
import logging

from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast, BertTokenizer

from typing import List


logger = logging.getLogger(__name__)


def parse_document(sentences: List[str], tokenizer: BertTokenizer,
                   max_seq_length=256):
    """
    Given a list of sentences in a document,
    generate examples for pre-training, all tokenized.
    """

    # First we flatten and tokenize the items
    all_tokens = [
        token for sentence in sentences
              for token in tokenizer.encode(
                  sentence, add_special_tokens=False)]

    # Then, we split in chunks based on max_seq_length, ignoring
    # Special tokens [CLS] and [SEP]
    target_seq_len = max_seq_length - 2

    chunks = [
        [tokenizer.cls_token_id] +
        all_tokens[start:start+target_seq_len] +
        [tokenizer.sep_token_id]
        for start in range(0, len(all_tokens), target_seq_len)]

    return chunks


def parse_file(file: str, max_seq_length: int, tokenizer: BertTokenizer):
    sentences = []
    examples = []

    with open(file, 'r', encoding='utf-8') as in_file:
        line = in_file.readline()

        while line:
            line = line.strip()

            if not line:  # end of document
                examples.extend(
                    parse_document(
                        sentences, tokenizer, max_seq_length))

                sentences = []
            else:
                sentences.append(line)

            line = in_file.readline()

    return examples


def from_datasetfiles(input_path, output_directory, tokenizer_name_or_path,
                      max_seq_length=512, max_examples=None):
    """
    """
    import os
    import random
    import pickle

    input_files = [str(f) for f in Path(input_path).glob('**/*')]
    
    # Using BERTTokenizerFast, which is a wrapper around a Rust
    # implementation, for handling performance issues due to large
    # dataset.
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name_or_path)

    logger.info(f'About to process {len(input_files)} dataset files.')

    for i, file in enumerate(tqdm(input_files)):
        examples = parse_file(file, max_seq_length, tokenizer)
        random.shuffle(examples)

        with open(os.path.join(output_directory, f'dataset_{i}.pkl'), 'wb+') as out_ds:
            pickle.dump(examples, out_ds)


def main():
    parser = argparse.ArgumentParser(description='Generate dataset examples '
                                     'from pre processed Wiki files '
                                     '(see preprocess.py).')

    parser.add_argument('input_dir', type=str,
                        help='the input directory with pre processed files')

    parser.add_argument('output_dir', type=str,
                        help='the directory where to store dataset files')

    parser.add_argument('--max_seq_length', metavar='n',
                        type=int, default=256,
                        help='the maximun length of a input sequence '
                             '(default=256)')

    parser.add_argument('--max_samples_per_file', metavar='n',
                        type=int, default=10000,
                        help='the maximum samples per file (default=10k)')

    args = parser.parse_args()


if __name__ == '__main__':
    main()
