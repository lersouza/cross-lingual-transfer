import argparse
import os

from tqdm import tqdm

from crosslangt.pretrain.dataset import create_training_intances
from pytorch_lightning import seed_everything
from tokenizers import BertWordPieceTokenizer


def generate_examples_from_file(file, vocab, output, max_seq_length):
    """
    Generate training examples from an specific file.
    """
    documents = [[]]
    tokenizer = BertWordPieceTokenizer(vocab, lowercase=False,
                                       strip_accents=False)

    all_lines = file.readlines()

    for line in tqdm(all_lines, f'loading {file.name}'):
        line = line.strip()

        if not line:  # End of document
            documents.append([])  # Prepare for the next doc
            continue

        # Tokenize line (sentence) and append to document
        encoded = tokenizer.encode(line, add_special_tokens=False)

        if encoded and encoded.tokens:
            documents[-1].append(encoded.tokens)

    output_path = os.path.join(output, f'{file.name}.examples')

    create_training_intances(
        documents, max_seq_length, output_path, tokenizer,
        tqdm_desc=f'creating instances for {file.name}')


def generate_examples(input_files, vocab, output, max_seq_length,
                      random_seed):
    """
    Generate training examples based on the `input_files`.
    """
    seed_everything(random_seed)  # Make it reproducible

    for file in input_files:
        with file:
            generate_examples_from_file(file, vocab, output, max_seq_length)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('input_files', type=argparse.FileType('r'), nargs='+',
                        help='Dataset files used to generate exampĺes.')

    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Max Sequence Length for BERT input_ids. '
                             'Default=512')

    parser.add_argument('-o', '--output', type=str, default='./',
                        help='Output directory to store examples')

    parser.add_argument('--vocab', type=str, default='vocab.txt',
                        help='The vocab file to tokenize input files.')

    parser.add_argument('--random_seed', type=int, default=54321,
                        help='A random seed to be able to reproduce results.')

    args = parser.parse_args()

    generate_examples(**vars(args))


if __name__ == '__main__':
    main()
