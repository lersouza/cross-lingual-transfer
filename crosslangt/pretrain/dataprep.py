import logging
import os
from pickle import HIGHEST_PROTOCOL
import torch

from pytorch_lightning import seed_everything
from random import random, randint, shuffle
from transformers import BertTokenizerFast
from tqdm import tqdm
from typing import List

logger = logging.getLogger(__name__)


def create_training_intances(documents,
                             max_seq_length,
                             output_path,
                             tokenizer: BertTokenizerFast,
                             use_tqdm=True,
                             tqdm_desc='processing documents'):
    """
    Creates Pre-training instances for the provided `documents`.
    This implementation is based on BERT's public code.

    Input parameters are:
    - documents: A list of documents. Each document is a list
                 of sentences and each sentence is a list of
                 WordPiece tokens.

    """
    logger.info(f'About to create instances for {len(documents)} docs.')

    instances = []

    # Make sure we don't have empty docs
    all_documents_ = [d for d in documents if d]
    all_docs_iterable_ = all_documents_  # Hack for optional tqdm

    # Do some shuffle, just to make sure the dataset will vary
    shuffle(all_documents_)

    if use_tqdm is True:
        all_docs_iterable_ = tqdm(all_documents_, desc=tqdm_desc)

    for i, document in enumerate(all_docs_iterable_):
        instances.extend(
            create_from_document(i, document, all_documents_, max_seq_length,
                                 tokenizer))

    logger.debug(f'Created {len(instances)} from document {i}')
    logger.info(f'Generated {len(instances)} instances'
                f'for {len(documents)} docs.')

    shuffle(instances)  # Shuffle for non-sequential doc sentences

    torch.save(instances, output_path, pickle_protocol=HIGHEST_PROTOCOL)

    return (output_path, len(instances))


def create_from_document(doc_idx, doc, all_docs, max_seq_length,
                         tokenizer: BertTokenizerFast):
    """
    I heavily rely on the implementation of BERT to generate training data:
    github.com/google-research/bert/blob/master/create_pretraining_data.py

    The main differences are:
    - I do not keep short sentences with any probability
    - The masking of tokens will be done dynamically, during training
      (just like the experiment made in RoBERTa paper)

    This function also assumes that all documents are tokenized
    (WordPiece Token Strings).
    """
    instances = []

    # Account for 1x [CLS] and 2x [SEP]
    target_seq_length = max_seq_length - 3

    # We'll use the same strategy as in the original
    # BERT paper, creating instances with a target max length
    # and using segments (groups of sentences) for that
    #
    # We create sentences pairs for next sentence prediction
    # where 50% of times the second sequence is the real next one.
    current_chunk = []
    current_length = 0

    i = 0  # A reference where we stopped in the current doc.

    while i < len(doc):
        segment = doc[i]
        current_chunk.append(segment)
        current_length += len(segment)

        if i == len(doc) - 1 or current_length >= target_seq_length:
            if current_chunk:
                sentence_a = []
                sentence_b = []

                sentence_a_end = randint(1, max(1, len(current_chunk) - 1))

                for ai in range(sentence_a_end):
                    sentence_a.extend(current_chunk[ai])

                is_next = True
                chance = random()

                if len(all_docs) > 1 and \
                   (len(current_chunk) == 1 or chance < 0.5):

                    sentence_b_tgt_len = target_seq_length - len(sentence_a)

                    # Let's get a random sentence
                    is_next = False

                    for _ in range(10):
                        random_doc_idx = randint(0, len(all_docs) - 1)

                        if random_doc_idx != doc_idx:
                            break

                    # We select the document and a random position to start
                    # We use len(random_doc) // 2 to make room for a bugger
                    # sentence
                    random_doc = all_docs[random_doc_idx]
                    random_start = randint(0, len(random_doc) // 2)

                    for j in range(random_start, len(random_doc)):
                        sentence_b.extend(random_doc[j])

                        if len(sentence_b) >= sentence_b_tgt_len:
                            break

                    # We free the tokens we'll not use for this instance
                    i -= len(current_chunk) - sentence_a_end
                else:
                    # It will be an actual next sentence
                    for j in range(sentence_a_end, len(current_chunk)):
                        sentence_b.extend(current_chunk[j])

                truncate_seq_pair(sentence_a, sentence_b, target_seq_length)

                assert len(sentence_a) >= 1
                assert len(sentence_b) >= 1

                final_seq = ['[CLS]'] + \
                    sentence_a + \
                    ['[SEP]'] + \
                    sentence_b + \
                    ['[SEP]']

                segment_ids = [0] * (len(sentence_a) + 2)
                segment_ids += [1] * (len(sentence_b) + 1)

                input_ids = tokenizer.convert_tokens_to_ids(final_seq)

                instances.append({
                    'input_ids': input_ids,
                    'token_type_ids': segment_ids,
                    'is_next': is_next
                })

            current_chunk = []
            current_length = 0
        i += 1
    return instances


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def generate_examples_from_file(file, tokenizer_name, output, max_seq_length):
    """
    Generate training examples from an specific file.
    """
    documents = [[]]
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_name)

    all_lines = file.readlines()
    file_name = os.path.split(file.name)[-1]  # file name only

    for line in tqdm(all_lines, f'loading {file_name}'):
        line = line.strip()

        if not line:  # End of document
            documents.append([])  # Prepare for the next doc
            continue

        # Tokenize line (sentence) and append to document
        encoded = tokenizer.tokenize(line)

        if encoded:
            documents[-1].append(encoded)

    output_path = os.path.join(output, f'{file_name}.examples')

    gen_file, num_examples = create_training_intances(
        documents,
        max_seq_length,
        output_path,
        tokenizer,
        tqdm_desc=f'creating instances for {file.name}')

    return gen_file, num_examples


def generate_examples(input_files,
                      tokenizer_name,
                      output,
                      max_seq_length,
                      random_seed,
                      files_type='train'):
    """
    Generate training examples based on the `input_files`.
    """
    seed_everything(random_seed)  # Make it reproducible
    os.makedirs(output, exist_ok=True)

    with open(os.path.join(output, f'{files_type}_index'), 'w+') as index:
        # First line is a configuration line
        # This line indicates the parameters for generating the data
        # {Max Sequence length}\t{Random Seed}\t{Tokenizer Name}
        # Then, we write an empty separator line.
        index.write(f'{max_seq_length}\t{random_seed}\t{tokenizer_name}\r\n')
        index.write('\r\n')

        for file in input_files:
            if type(file) is str:
                file = open(file)

            with file:
                gen, qty = generate_examples_from_file(file, tokenizer_name,
                                                       output, max_seq_length)

                gen = os.path.abspath(gen)  # resolve the full path

                index.write(f'{gen}\t{qty}\r\n')
