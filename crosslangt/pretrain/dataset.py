import linecache
import math
import os
import torch

from dataclasses import dataclass
from itertools import chain
from torch.utils.data.dataset import IterableDataset
from transformers.tokenization_bert import BertTokenizer


@dataclass
class IndexEntry:
    start_index: int
    end_index: int
    file_location: str


class LexicalTrainDataset(IterableDataset):
    def __init__(self, index_file: str, tokenizer: BertTokenizer,
                 mlm_probability: float = 0.15, max_examples: int = None):

        assert os.path.exists(index_file)

        self.index = []
        self.total_examples = 0
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.max_examples = max_examples

        start_index = 0

        with open(index_file, "r") as index:
            for line in index.readlines():
                file_name, num_examples = line.split("\t")
                num_examples = int(num_examples.strip())

                self.index.append(
                    IndexEntry(
                        start_index,
                        (start_index + num_examples - 1),
                        file_name.strip())
                )

                start_index += num_examples
                self.total_examples += num_examples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single worker... we process all entries sequentially
            return chain.from_iterable(map(self.__process_entry, self.index))
        else:
            # Multiple workers in context
            # Well partition the index entries per worker.

            num_workers = worker_info.num_workers
            per_worker = int(math.ceil(len(self.index) / float(num_workers)))
            worker_id = worker_info.id

            start_entry = worker_id * per_worker
            end_entry = min(start_entry + per_worker, len(self.index))

            worker_subset = self.index[start_entry:end_entry]

            return chain.from_iterable(
                map(self.__process_entry, worker_subset))

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

    def __process_entry(self, index_entry: IndexEntry):
        with open(index_entry.file_location, 'r', encoding='utf-8') as dfile:
            for line_num, line in enumerate(dfile):
                if self.__proceed(index_entry, line_num) is True:
                    yield self.__convert_dataset_file_entry(line)

    def __proceed(self, index_entry: IndexEntry, actual_file_line_number: int):
        global_line_number = index_entry.start_index + actual_file_line_number
        return global_line_number < self.max_examples

    def __convert_dataset_file_entry(self, dataset_file_entry: str):
        raw_data_parts = dataset_file_entry.strip().split("\t")

        input_ids = [int(item) for item in raw_data_parts[0].split()]
        token_type_ids = [int(item) for item in raw_data_parts[1].split()]
        label = int(raw_data_parts[2])

        input_ids = torch.tensor(input_ids)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": torch.tensor(token_type_ids),
            "next_sentence_label": torch.tensor(label),
        }
