import os
import lxml

from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from typing import List


@dataclass
class NLIExample:
    """ Represents and example for NLI tasks. """
    pairID: int
    originalIndex: int
    sentence_a: str
    sentence_b: str
    label: str


class NLIDataset(Dataset):
    """
    This class holds data for NLI tasks in NLP models.

    It provides a __getitem__ implementation which captures the necessary information
    for model training: first_sequence, second_sequence and labels (all encoded).
    """
    def __init__(self, 
                 examples: List[NLIExample],
                 tokenizer: PreTrainedTokenizer,
                 max_seq_length: int = 512,
                 labels: List[str] = ['neutral', 'entailment', 'contradiction']):
        self.examples = examples
        self.tokenizer = tokenizer
        self.labels = labels

        self.max_seq_length = max_seq_length

    def __len__(self):
        """ Returns the size of the dataset. """
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Returns an example indexed by idx.

        The returned value is a tuple with the following:
        input_ids ([0]): the sentence pair encoded with the tokenizer.
        attention_mask ([1]): the attention mask for the encoded input.
        token_type_ids ([2]): the token types for the input sequence.
        label_idx ([3]): the label index for the example (based on init labels)
        """
        example = self.examples[idx]

        tokenized = self.tokenizer.encode_plus(
            example.sentence_a, example.sentence_b,
            max_length=self.max_seq_length,
            pad_to_max_length=True, return_token_type_ids=True,
            return_tensors='pt')

        labels_idx = self.labels.index(example.label)

        return (
            tokenized['input_ids'],
            tokenized['attention_mask'],
            tokenized['token_type_ids'],
            labels_idx
        )


MNLI_PAIRID_INDEX = 2
MNLI_SEQ1_INDEX = 8
MNLI_SEQ2_INDEX = 9
MNLI_LABEL_INDEX = -1


def load_mnli_dataset(root_path: str, file_name: str,
                      tokenizer: PreTrainedTokenizer,
                      max_seq_length: int = 512):
    """
    Retrieves a NLIDataset from a MNLI file.

    root_path: The path where MNLI files are located.
    file_name: The name of the file to load.
    tokenizer: The tokenizer to use when encoding examples.
    """
    full_path = os.path.join(root_path, file_name)

    if not os.path.exists(full_path):
        raise ValueError(f'The file {full_path} does not exist.')

    examples = []

    with open(full_path, 'r', encoding='utf-8') as tsv_file:
        file_lines = tsv_file.readlines()

        for i, line in enumerate(file_lines, 1): # Skipping headers
            if line:
                examples.append(parse_mnli_sample(i, line.strip()))

    return NLIDataset(examples, tokenizer, max_seq_length=max_seq_length)


def parse_mnli_sample(row_index, raw_sample) -> NLIExample:
    """ Parses a MNLI raw line, in TSV format. """
    sample_parts = raw_sample.split('\t')

    example = NLIExample(
        sample_parts[MNLI_PAIRID_INDEX],
        row_index,
        sample_parts[MNLI_SEQ1_INDEX], 
        sample_parts[MNLI_SEQ2_INDEX],
        sample_parts[MNLI_LABEL_INDEX])

    return example
    

def load_assin_dataset(root_path, file_name):
    pass
