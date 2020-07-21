import logging
import torch

from torch import Tensor
from torch.nn.modules.sparse import Embedding
from transformers import BertModel, BertTokenizer


logger = logging.getLogger(__name__)

training_lexical_strategies = ['freeze-all',
                               'freeze-special',
                               'freeze-nonspecial',
                               'none']

testing_lexical_strategies = ['original',
                              'target-lexical-original-special',
                              'target-lexical']


def setup_lexical_for_training(strategy,
                               model: BertModel,
                               tokenizer: BertTokenizer):
    """ Setups the lexical part of `model` based on `strategy`. """
    assert strategy in training_lexical_strategies

    original_embeddings = model.get_input_embeddings()
    tobe_embeddings = None

    # We cut on the last kwown special token
    # So we get the latest index + 1
    # (for instance, if the last one is 103, we get 104 ).
    # This is because SlicedEmbeddings cut on [:cut].
    bert_special_tokens_cut = sorted(tokenizer.all_special_ids)[-1] + 1

    if strategy == 'freeze-special':
        tobe_embeddings = SlicedEmbedding.slice(
            original_embeddings,
            bert_special_tokens_cut,
            freeze_first_part=True,
            freeze_second_part=False)
    elif strategy == 'freeze-nonspecial':
        tobe_embeddings = SlicedEmbedding.slice(
            original_embeddings,
            bert_special_tokens_cut,
            freeze_first_part=False,
            freeze_second_part=True)
    elif strategy == 'freeze-all':
        tobe_embeddings = original_embeddings
        tobe_embeddings.weight.requires_grad = False
    else:
        tobe_embeddings = original_embeddings

    model.set_input_embeddings(tobe_embeddings)


def setup_lexical_for_testing(strategy: str,
                              model: BertModel,
                              tokenizer: BertTokenizer,
                              target_lexical: str):
    """
    Setup the model lexical part according to `strategy`.
    Strategy can be:

    - original: The model's lexical will be used as is.
    - target_lexical_full: The lexical located at `target_lexical`
                           will be used in model\'s, including special tokens.
    - target_lexical_keep_special: The lexical located at `target_lexical`
                                   will be used, but the model\'s
                                   embeddings for special tokens
                                   will be preserved.
    """
    assert strategy in testing_lexical_strategies
    assert model is not None
    assert tokenizer is not None

    if strategy == 'original':
        return  # Nothing to do here

    target_lexical_state = torch.load(target_lexical)

    if strategy == 'target-lexical':
        model.set_input_embeddings(
            new_like(model.get_input_embeddings(), target_lexical_state)
        )
    elif strategy == 'target-lexical-original-special':
        assert model.get_input_embeddings().embedding_dim == \
            target_lexical_state['weight'].shape[1]

        # We cut on the last kwown special token
        # So we get the latest index + 1
        # (for instance, if the last one is 103, we get 104 ).
        # This is because SlicedEmbeddings cut on [:cut].
        bert_special_tokens_cut = sorted(tokenizer.all_special_ids)[-1] + 1

        model_weights = model.get_input_embeddings().weight
        target_weights = target_lexical_state['weight']

        tobe = SlicedEmbedding(model_weights[:bert_special_tokens_cut],
                               target_weights[bert_special_tokens_cut:],
                               True, True)  # For testing, both are freezed

        model.set_input_embeddings(tobe)
    else:
        raise NotImplementedError(f'strategy {strategy} is not implemented')


def new_like(base_embedding: Embedding, state_dict: dict):
    """
    Creates a new embedding with state `state_dict`, but with `e` parameters.
    """
    assert 'weight' in state_dict

    vocab_size = state_dict['weight'].shape[0]
    embedding_dim = state_dict['weight'].shape[1]

    clone = Embedding(vocab_size, embedding_dim,
                      padding_idx=base_embedding.padding_idx,
                      max_norm=base_embedding.max_norm,
                      norm_type=base_embedding.norm_type,
                      scale_grad_by_freq=base_embedding.scale_grad_by_freq,
                      sparse=base_embedding.sparse)

    clone.load_state_dict(state_dict)

    return clone


class SlicedEmbedding(torch.nn.Module):
    """
    This module implements an Embedding capable of partial freezing.
    The idea is to slice the original Embedding in 2 and make freezing
    individual slices independently.

    Inspired by:
    - https://discuss.pytorch.org/t/partially-freeze-embedding-layer/18458/7
    - https://stackoverflow.com/questions/54924582/is-it-possible-to-freeze-
      only-certain-embedding-weights-in-the-embedding-layer-i
    """
    def __init__(self,
                 weigths_a: torch.Tensor,
                 weigths_b: torch.Tensor,
                 freeze_a: bool = False,
                 freeze_b: bool = False,
                 **kwargs):
        """
        Initializes the Embedding Module.

        Parameters:
        - weigths_a: The weigths of the first embedding part.
        - weigths_b: The weigths of the second embedding part.
        - freeze_a: Indicates whether `weights_a` should be freezed.
        - freeze_b: Indicates whether `weights_b` should be freezed.
        - kwargs: Arguments to be passed to Embedding.from_pretrained
        """
        assert weigths_a.shape[1] == weigths_b.shape[1]
        assert type(freeze_a) is bool
        assert type(freeze_b) is bool

        super(SlicedEmbedding, self).__init__()

        self.first_embedding = torch.nn.Embedding.from_pretrained(
            weigths_a, freeze=freeze_a, **kwargs)

        self.second_embedding = torch.nn.Embedding.from_pretrained(
            weigths_b, freeze=freeze_b, **kwargs)

        self.embedding_dim = weigths_a.shape[1]

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

    @staticmethod
    def slice(embedding: torch.nn.Embedding, slice_cut: int,
              freeze_first_part: bool, freeze_second_part: bool):
        """
        Slices an Embedding, enabling the freezing of some positions.

        The final embedding will be composed by two:
        - fisrt: embedding.weight[:slice_cut]
        - second: embedding.weight[slice_cut:]
        """
        weigths_a = embedding.weight[:slice_cut]
        weigths_b = embedding.weight[slice_cut:]

        scale_grad_by_freq = embedding.scale_grad_by_freq

        sliced = SlicedEmbedding(weigths_a, weigths_b,
                                 freeze_first_part, freeze_second_part,
                                 padding_idx=embedding.padding_idx,
                                 max_norm=embedding.max_norm,
                                 norm_type=embedding.norm_type,
                                 sparse=embedding.sparse,
                                 scale_grad_by_freq=scale_grad_by_freq)

        return sliced
