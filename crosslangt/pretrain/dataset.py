from torch.utils.data import Dataset


class WikiMaskedDataset(Dataset):
    """
    This dataset holds wikipedia data for MLM pre-training objective.
    """
    pass


class WikiSentencePredictionDataset(Dataset):
    """
    This dataset holds data Wikipedia data for Next Sentence prediction
    pre training objective.
    """
    pass


def load_wiki_data(input_dir):
    """
    Loads the Wikipedia data extracted from Wikiextractor tool
    into lists of sentences.
    """
    pass
