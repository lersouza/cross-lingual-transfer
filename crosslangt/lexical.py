import logging
import torch


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

    embeddings.load_state_dict(weights)

    model.set_input_embeddings(embeddings)