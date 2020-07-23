import linecache
import os
from typing import Tuple
import torch

from dataclasses import dataclass
from torch.utils.data import Dataset
from transformers.tokenization_bert import BertTokenizer


@dataclass
class IndexEntry:
    start_index: int
    end_index: int
    file_location: str


class LexicalTrainDataset(Dataset):
    def __init__(self, index_file: str, tokenizer: BertTokenizer,
                 mlm_probability: float = 0.15):
        assert os.path.exists(index_file)

        self.index = []
        self.total_examples = 0
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

        start_index = 0

        with open(index_file, "r") as index:
            for line in index.readlines():
                file, num_examples = line.split("\t")
                num_examples = int(num_examples.strip())

                self.index.append(
                    IndexEntry(
                        start_index,
                        (start_index + num_examples - 1),
                        file)
                )

                start_index += num_examples
                self.total_examples += num_examples

    def __len__(self):
        return self.total_examples

    def __getitem__(self, index: int):
        return self.__resolve(index)

    def collate_batch(self, batch):
        elem = batch[0]

        collated = {key: torch.stack([d[key] for d in batch]) for key in elem}
        masked_inputs, mask_labels = self.mask_tokens(collated['input_ids'])

        collated['input_ids'] = masked_inputs
        collated['mlm_labels'] = mask_labels

        return collated

    def mask_tokens(self, inputs: torch.Tensor):
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
                "flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(
                val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix.masked_fill_(
            torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0
        )
        if self.tokenizer._pad_token is not None:
            padding_mask = labels.eq(self.tokenizer.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with
        # tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(
                torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input
        # tokens unchanged
        return inputs, labels

    def __resolve(self, index: int):
        entry = self.__find_entry(index)
        line_in_file = (index - entry.start_index) + 1

        raw_data = linecache.getline(entry.file_location, line_in_file)
        raw_data_parts = raw_data.strip().split("\t")

        input_ids = [int(item) for item in raw_data_parts[0].split()]
        token_type_ids = [int(item) for item in raw_data_parts[1].split()]
        label = int(raw_data_parts[2])

        input_ids = torch.tensor(input_ids)
        attention_mask = (input_ids != 0).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": torch.tensor(token_type_ids),
            "next_sentence_label": torch.tensor(label),
        }

    def __find_entry(self, index: int):
        actual_entry = None

        for entry in self.index:
            if index >= entry.start_index and index <= entry.end_index:
                actual_entry = entry
                break

        if actual_entry is None:
            raise IndexError(f"Could not find index {index}")

        return actual_entry
