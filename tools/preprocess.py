import argparse
import json
import logging
import nltk
import os

from tqdm import tqdm


logger = logging.getLogger(__name__)


def load_documents(file_path, sent_tokenizer: nltk.PunktSentenceTokenizer):
    """ Loads a file as a set of documents, each represented as a list of lines. """
    documents = []
    with open(os.path.join(file_path), encoding='utf-8') as f:
        for line in f.read().splitlines():
            if line.startswith('<doc'):
                documents.append([])
            elif line == '</doc>': continue
            elif not line.strip(): continue
            elif line[:2] == '[[': continue
            else:
                if sent_tokenizer:
                    line_sentences = sent_tokenizer.tokenize(line)
                    documents[-1].extend(line_sentences)
                else:
                    print('not with tokenizer:', sent_tokenizer)
                    documents[-1].append(line)


    return documents


def parse_raw_document(document: dict, min_sentence_length):
    """ Parses a document string (JSON format) to a list of sentences. """
    document_url = document['url']

    logger.debug(f'Parsing document {document_url}')

    doc_text = document['text']
    paragraphs = document['text'].splitlines()

    # We consider each line in the extracted Wiki Dunp as a sentence.
    # This seems to be a good decision, so no other processing is done,
    # preserving the original article.
    # In the original paper, the authors define a sentence as:
    #       "an arbitrary span of contiguous text,
    #           rather than an actual linguistic sentence."
    sentences_in_doc = [seq for seq in doc_text.splitlines()
                        if seq and not seq.isspace()
                        and len(seq) > min_sentence_length]

    logger.debug(f'Found {len(sentences_in_doc)} in {document_url}.')

    return sentences_in_doc


def find_files_to_process(input_path):
    """ This function returns a list with all files to be processed. """
    all_files = []

    for root, _, files in os.walk(input_path):
        all_files.extend([os.path.join(root, f) for f in files])

    return all_files


def write_pre_processed(target_dir, source_file, documents):
    """
    Writes parsed documents to a target file, one sentence per line,
    two blank lines to indicate document end.

    The target file will be under target_dir, following the structure:
    <target_dir>/<source_file>
    """
    output_path, file_name = os.path.split(source_file)
    output_path = os.path.join(target_dir, output_path)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    target_file_path = os.path.join(output_path, file_name)

    with open(target_file_path, 'w+', encoding='utf-8') as file:
        for document in documents:
            for sentence in document:
                file.write(f'{sentence}\n')

            file.write('\n\n')  # Separate documents with two blank lines


def process_data(input_path, output_path,
                 min_sentence_length, split_sentence_lang):
    """
    Pre Process documents and their sentences from WikiExtractor files.
    The result is, then, saved to output_path.

    Parameters:
    - input_path: the path where files extracted are located.
    - output_path: the path where to store pre-processed results.
    - min_sentence_length: the minimum len of a sentence to be considered

    """
    assert os.path.exists(input_path)
    assert os.path.isdir(input_path)
    assert os.path.exists(output_path)

    files_to_process = find_files_to_process(input_path)
    logger.info(f'{len(files_to_process)} files found to be processed.')

    sent_tokenizer = None

    if split_sentence_lang:
        nltk.download('punkt')
        sent_tokenizer = nltk.data.load(
            f'tokenizers/punkt/{split_sentence_lang}.pickle')

    progress_bar = tqdm(files_to_process)

    for file in progress_bar:
        progress_bar.set_description(file)

        documents = load_documents(file, sent_tokenizer)
        write_pre_processed(output_path, file, documents)


def main():
    """ Executes the preprocessor utility. """
    parser = argparse.ArgumentParser(
        description='Pre process tool for Wiki Dump dataset, '
                    'extracted by WikiExtractor')

    parser.add_argument('input_path',
                        help='the path where files processed by '
                             'WikiExtractor are located.')

    parser.add_argument('output_path',
                        help='a path where pre processed files '
                             'in txt format will be saved.')

    parser.add_argument('--min_length',
                        type=int, default=0,
                        metavar='M',
                        help='minimum size of a sentence to keep it. '
                             'Default=0')

    parser.add_argument('--split_sentence_lang',
                        type=str, default=None,
                        metavar='LANG',
                        help='if provided, uses nltk Pukt tokenizer'
                             'indicated in file "tokenizers/punkt/<LANG>.pickle')

    parser.add_argument('--debug',
                        action='store_true',
                        help='enabled debug logging.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    process_data(args.input_path, args.output_path, args.min_length,
                 args.split_sentence_lang)


if __name__ == '__main__':
    main()
