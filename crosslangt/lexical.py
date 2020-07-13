import logging
import torch

from torch import Tensor
from transformers import BertTokenizer


logger = logging.getLogger(__name__)


def freeze_lexical(model):
    """ Freezes the lexical part of Bert Model. """
    logger.info('BERT-MNLI: Freezing BERT model lexical. '
                'All Input Embeddings will not be updated.')

    embeddings = model.get_input_embeddings()

    for parameter in embeddings.parameters():
        parameter.requires_grad = False


def load_pretrained_lexical(model, lexical):
    """ Loads a pre-trained lexical layer specified in lexical. """
    lexical_state = torch.load(lexical)
    weights = lexical_state['weight']

    embeddings = torch.nn.Embedding(
        weights.shape[0], weights.shape[1],
        padding_idx=0)

    embeddings.load_state_dict(lexical_state)

    model.set_input_embeddings(embeddings)


def get_tokenizer_from_vocab(vocab_file, lowercase=False):
    """ Retrieves a `PreTrainedTokenizer` from a vocab file. """
    tokenizer = BertTokenizer(
        vocab_file, do_lower_case=lowercase)

    return tokenizer


class SlicedEmbeddings(torch.nn.Module):
    """
    This module implements an Embedding capable of partial freezing.
    The idea is to slice the original Embedding in 2 and make freezing
    individual slices in an independently.

    Inspired by:
    - https://discuss.pytorch.org/t/partially-freeze-embedding-layer/18458/7
    - https://stackoverflow.com/questions/54924582/is-it-possible-to-freeze-
      only-certain-embedding-weights-in-the-embedding-layer-i
    """
    def __init__(self,
                 original: torch.nn.Embedding,
                 slice_cut: int,
                 freeze_first: bool = False,
                 freeze_second: bool = False):
        """
        Initializes the Embedding Module.

        Parameters:
        - original: The original Embeddings for starting point.
        - slice_cut: The index where to slice.
                     First embedding will be original[:slice_cut].
                     Second will be original[slice_cut:]
        - freeze_first: Indicates whether first slice must be freezed.
        - freeze_second: Indicates whether second slice must be freezed.
        """
        assert original is not None
        assert slice_cut <= original.num_embeddings
        assert type(freeze_first) is bool

        super(SlicedEmbeddings, self).__init__()

        first_slice = original.weight[0: slice_cut]
        second_slice = original.weight[slice_cut:]

        self.first_embedding = torch.nn.Embedding.from_pretrained(
            first_slice, freeze=freeze_first,
            padding_idx=original.padding_idx,
            max_norm=original.max_norm, norm_type=original.norm_type,
            scale_grad_by_freq=original.scale_grad_by_freq,
            sparse=original.sparse
        )

        self.second_embedding = torch.nn.Embedding.from_pretrained(
            second_slice, freeze=freeze_second,
            padding_idx=original.padding_idx,
            max_norm=original.max_norm, norm_type=original.norm_type,
            scale_grad_by_freq=original.scale_grad_by_freq,
            sparse=original.sparse
        )

        self.embedding_dim = original.embedding_dim

    def forward(self, batch: Tensor):
        """
        Lookup indexes from `batch` in internal Embeddings table.
        """
        mask = batch >= self.first_embedding.num_embeddings

        second_embedding_ids = batch - self.first_embedding.num_embeddings
        second_embedding_ids[~mask] = 0

        batch = batch.masked_fill(mask, 0)

        embeddings = self.first_embedding(batch)
        embeddings[mask] = self.second_embedding(second_embedding_ids)[mask]

        return embeddings
