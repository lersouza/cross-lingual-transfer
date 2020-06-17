import logging

from dataclasses import dataclass
from random import random, randrange, randint
from tokenizers import BertWordPieceTokenizer
from typing import List


logger = logging.getLogger(__name__)


@dataclass
class PreTrainInstace():
    sentence_pair: List[str]
    segment_ids: List[int]
    is_next: bool


def create_training_intances(documents, max_seq_length,
                             tokenizer: BertWordPieceTokenizer):
    logger.info(f'About to create instances for {len(documents)} docs.')

    instances = []

    for i, document in enumerate(documents):
        instances.extend(create_from_document(
            i, document, documents, max_seq_length, tokenizer))

        logger.debug(f'Created {len(instances)} from document {i}')

    logger.info(f'Generated {len(instances)} instances'
                f'for {len(documents)} docs.')

    return instances


def create_from_document(doc_idx, doc, all_docs,
                         max_seq_length,
                         tokenizer: BertWordPieceTokenizer):
    """
    I heavily rely on the implementation of BERT to generate training data:
    github.com/google-research/bert/blob/master/create_pretraining_data.py

    The main differences are:
    - I do not keep short sentences with any probability
    - The masking of tokens will be done dynamically, during training
      (just like the experiment made in RoBERTa paper)
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
            sentence_a_end = randrange(max(1, len(current_chunk) - 1))

            sentence_a = current_chunk[0:sentence_a_end]
            sentence_b = []

            is_next = True
            sentence_b_tgt_len = target_seq_length - len(sentence_a)

            if random() < 0.5:
                # Let's get a random sentence
                is_next = False
                random_doc_idx = randint(0, len(all_docs) - 1)

                if random_doc_idx == doc_idx:
                    random_doc -= 1

                random_doc = all_docs[random_doc_idx]
                random_start = randint(0, len(random_doc) - 1)

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

            final_seq = ['[CLS]'] + \
                sentence_a + \
                ['[SEP]'] + \
                sentence_b + \
                ['[SEP]']

            segment_ids = [0] * (len(sentence_a) + 2)
            segment_ids += [1] * (len(sentence_b) + 1)

            instances.append(PreTrainInstace(final_seq, segment_ids, is_next))

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
