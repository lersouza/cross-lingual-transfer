from argparse import ArgumentParser, FileType
from crosslangt.pretrain.dataprep import generate_examples


def main():
    parser = ArgumentParser()

    parser.add_argument('input_files', type=FileType('r'), nargs='+',
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

    parser.add_argument('--files_type', type=str, default='train',
                        help='The type of files: train, eval, test.')

    args = parser.parse_args()

    generate_examples(**vars(args))


if __name__ == '__main__':
    main()
