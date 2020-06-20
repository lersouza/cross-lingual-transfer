from transformers import BertTokenizerFast
from tokenizers import BertWordPieceTokenizer


def get_tokenizer(model, vocab_file, type='transformers'):
    if vocab_file:
        return tokenizer_from_vocab(vocab_file, type=type)

    return BertTokenizerFast.from_pretrained(model)


def tokenizer_from_vocab(vocab_file: str, type='transformers'):
    if type == 'transformers':
        return BertTokenizerFast(
            vocab_file, do_lower_case=False, strip_accents=False)
    else:
        return BertWordPieceTokenizer(vocab_file, strip_accents=False,
                                      lowercase=False)
