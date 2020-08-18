from unittest.mock import MagicMock


def get_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode_plus = MagicMock()
    tokenizer.encode_plus.return_value = {
        'input_ids': [1, 2],
        'attention_mask': [1, 1],
        'token_type_ids': [0, 0]
    }

    return tokenizer
