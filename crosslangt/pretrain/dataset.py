import linecache
import math
import os
import torch

from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


@dataclass
class IndexEntry:
    start_index: int
    end_index: int
    file_location: str


class LexicalTrainDataset(Dataset):
    """
    Represents a dataset used by aligning Lexical of a model.
    This datatset contains a collate batch utility that can be used
    to dynamically Mask tokens, as in RoBERTa training.
    """
    def __init__(self,
                 index_file: str,
                 tokenizer: PreTrainedTokenizer,
                 mlm_probability: float = 0.15,
                 max_examples: int = None,
                 max_seq_length: int = None):

        assert os.path.exists(index_file)

        self.index = []
        self.index_config = {}
        self.examples = []

        # Keep track of all examples referenced by index_file
        # But, since user can limit the number of examples,
        # use __len__ to get the actual number of loaded instances
        self.total_examples = 0
        self.max_examples = max_examples
        self.override_max_seq_length = max_seq_length

        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

        self._build_index(index_file)
        self._load_examples()

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        pad_token_id = self.tokenizer.pad_token_id
        max_seq_length = self.override_max_seq_length or self.index_config.get(
            'max_seq_length', 256)

        input_ids = torch.empty(max_seq_length).long().fill_(pad_token_id)
        token_type_ids = torch.empty(max_seq_length).long().fill_(pad_token_id)

        e_input_ids = torch.tensor(example['input_ids'], dtype=torch.long)
        e_type_ids = torch.tensor(example['token_type_ids'], dtype=torch.long)

        target_length = min(len(e_input_ids), max_seq_length)

        input_ids[:target_length] = e_input_ids[:target_length]
        token_type_ids[:target_length] = e_type_ids[:target_length]

        is_next = torch.tensor(example['is_next']).long()

        attention_mask = torch.zeros_like(input_ids)
        attention_mask.masked_fill_(input_ids != pad_token_id, 1)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'next_sentence_label': is_next
        }

    def collate_batch(self, batch):
        elem = batch[0]

        collated = {key: torch.stack([d[key] for d in batch]) for key in elem}
        masked_inputs, mask_labels = self._mask_tokens(collated['input_ids'])

        collated['input_ids'] = masked_inputs
        collated['mlm_labels'] = mask_labels

        return collated

    def _mask_tokens(self, inputs: torch.Tensor):
        """
        Extracted from Huggingface DataCollatorForLanguageModeling

        Prepare masked tokens inputs/labels for masked language modeling:
        - 80% MASK
        - 10% random
        - 10% original
        """
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is "
                "necessary for masked language modeling. Remove the --mlm "
                "flag if you want to use this tokenizer.")

        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(
                val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask,
                                                     dtype=torch.bool),
                                        value=0.0)
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with
        # tokenizer.mask_token ([MASK])
        indices_replaced = (torch.bernoulli(torch.full(
            labels.shape, 0.8)).bool() & masked_indices)
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced)
        random_words = torch.randint(len(self.tokenizer),
                                     labels.shape,
                                     dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input
        # tokens unchanged
        return inputs, labels

    def _build_index(self, index_file):
        start_index = 0

        with open(index_file, "r") as index:
            index_lines = index.readlines()

            # First line is a config line
            seq_length, seed, tokenizer = index_lines[0].strip().split("\t")

            self.index_config['max_seq_length'] = int(seq_length)
            self.index_config['data_seed'] = int(seed)
            self.index_config['data_tokenizer'] = tokenizer

            # We skip the first config line and the next empty, sperator line
            for line_idx in range(2, len(index_lines)):
                line = index_lines[line_idx]

                file_name, num_examples = line.split("\t")
                num_examples = int(num_examples.strip())

                self.index.append(
                    IndexEntry(start_index, (start_index + num_examples - 1),
                               file_name.strip()))

                start_index += num_examples
                self.total_examples += num_examples

        assert len(self.index) > 0

    def _load_examples(self):
        self.examples = []

        for entry in self.index:
            dataset_examples = torch.load(entry.file_location)
            self.examples.extend(dataset_examples)

            if self.max_examples and len(self.examples) >= self.max_examples:
                break

        if self.max_examples:
            self.examples = self.examples[:self.max_examples]
