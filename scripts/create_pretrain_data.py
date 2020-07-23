import argparse
import os
from os import name

from tqdm import tqdm
from transformers.tokenization_bert import BertTokenizerFast

from crosslangt.pretrain.dataprep import create_training_intances
from pytorch_lightning import seed_everything


def generate_examples_from_file(file, tokenizer_name, output, max_seq_length):
    """
    Generate training examples from an specific file.
    """
    documents = [[]]
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)

    all_lines = file.readlines()

    for line in tqdm(all_lines, f'loading {file.name}'):
        line = line.strip()

        if not line:  # End of document
            documents.append([])  # Prepare for the next doc
            continue

        # Tokenize line (sentence) and append to document
        encoded = tokenizer.tokenize(line)

        if encoded:
            documents[-1].append(encoded)

    output_path = os.path.join(output, f'{file.name}.examples')

    gen_file, num_examples = create_training_intances(
        documents, max_seq_length, output_path, tokenizer,
        tqdm_desc=f'creating instances for {file.name}')

    return gen_file, num_examples


def generate_examples(input_files, tokenizer_name, output, max_seq_length,
                      random_seed):
    """
    Generate training examples based on the `input_files`.
    """
    seed_everything(random_seed)  # Make it reproducible

    with open(os.path.join(output, 'index_file'), 'w+') as index:
        for file in input_files:
            with file:
                gen, qty = generate_examples_from_file(
                    file, tokenizer_name, output, max_seq_length)

                gen = os.path.abspath(gen)  # resolve the full path

                index.write(f'{gen}\t{qty}\r\n')


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('input_files', type=argparse.FileType('r'), nargs='+',
                        help='Dataset files used to generate exampĺes.')

    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='Max Sequence Length for BERT input_ids. '
                             'Default=512')

    parser.add_argument('-o', '--output', type=str, default='./',
                        help='Output directory to store examples')

    parser.add_argument('--tokenizer_name', type=str, default='vocab.txt',
                        help='The name of the tokenizer to use.')

    parser.add_argument('--random_seed', type=int, default=123,
                        help='A random seed to be able to reproduce results.')

    args = parser.parse_args()

    generate_examples(**vars(args))


if __name__ == '__main__':
    main()
