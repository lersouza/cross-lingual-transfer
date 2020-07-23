import h5py
import logging
import numpy as np

from dataclasses import dataclass
from random import random, randint, shuffle
import torch
from transformers import BertTokenizerFast
from tqdm import tqdm
from typing import List


logger = logging.getLogger(__name__)


@dataclass
class PreTrainInstace():
    """
    Represents and example (instance) for Pre training BERT.
    This examples consists of:

    - sentence_pair: A list of tokens representing a sentence pair.
                     Start token is always [CLS] and end token is [SEP].
                     The sentences are divided by an additional [SEP] token.

                     Example:
                     - '[CLS]', 'sent1', '[SEP]', 'sent2', '[SEP]'

    - segment_ids:   A list of size len(sentence_pair) with 0 to represent
                     the tokens of first sentence and 1 for the second.

    - is_next:       A flag indicating whether or not the second sentence
                     actual follows the first one. Used for NSP task.
    """
    sentence_pair: List[str]
    segment_ids: List[int]
    is_next: bool


def create_training_intances(documents, max_seq_length, output_path,
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
        instances.extend(create_from_document(
            i, document, all_documents_, max_seq_length))

    logger.debug(f'Created {len(instances)} from document {i}')
    logger.info(f'Generated {len(instances)} instances'
                f'for {len(documents)} docs.')

    shuffle(instances)  # Shuffle for non-sequential doc sentences

    return write_instances_to_file(
        instances, max_seq_length, output_path, tokenizer)


def write_instances_to_file(
    instances: List[PreTrainInstace], max_seq_length: int, output_path: str,
        tokenizer: BertTokenizerFast):

    with open(output_path, mode='w') as sfile:
        for i, instance in enumerate(tqdm(instances, 'saving instances')):
            seq_size = len(instance.sentence_pair)
            input_ids = tokenizer.convert_tokens_to_ids(instance.sentence_pair)
            padding_size = max_seq_length - seq_size

            input_ids = input_ids + [0] * padding_size
            segment_ids = instance.segment_ids + [0] * padding_size
            label = 1 if instance.is_next else 0

            in_str = ' '.join([str(in_id) for in_id in input_ids])
            tok_ty_str = ' '.join([str(in_id) for in_id in segment_ids])

            sfile.write(f'{in_str}\t{tok_ty_str}\t{label}\r\n')

    return (output_path, i+1)


def create_from_document(doc_idx, doc, all_docs, max_seq_length):
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

                instances.append(
                    PreTrainInstace(
                        sentence_pair=final_seq,
                        segment_ids=segment_ids,
                        is_next=is_next))

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
