import torch

from typing import List
from dataclasses import dataclass
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class NLIExample():
    pairID: str
    label: List[int]
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int] = None


class NLIDataset(Dataset):
    def __init__(self, examples: List[NLIExample]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        features = {
            'input_ids': torch.tensor(example.input_ids),
            'attention_mask': torch.tensor(example.attention_mask),
            'label': torch.tensor(example.label)
        }

        if example.token_type_ids is not None:
            features['token_type_ids'] = torch.tensor(example.token_type_ids)

        return features
