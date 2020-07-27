from os import pathconf
from unittest import TestCase
from unittest.mock import MagicMock, patch

from crosslangt.pretrain.tokenizer import (train_tokenizer,
                                           get_bert_initial_alphabet)


class TrainTokenizerTestCase(TestCase):

    @patch('crosslangt.pretrain.tokenizer.json')
    def test_train_tokenizer(self, json_mock):
        files = ['file1', 'file2']
        initial_vocab = get_bert_initial_alphabet()

        tokenzer_type = 'crosslangt.pretrain.tokenizer.BertWordPieceTokenizer'
        json_mock_type = 'crosslangt.pretrain.tokenizer.json.dumps'

        tokenizer_mock = MagicMock()
        json_mock = MagicMock()

        with patch(tokenzer_type, return_value=tokenizer_mock):
            with patch(json_mock_type):
                train_tokenizer(
                    files,
                    'my_tokenizer',
                    '/tmp',
                    30_000,
                    False,
                    False)

            tokenizer_mock.train.assert_called_with(
                files,
                initial_alphabet=initial_vocab,
                vocab_size=30_000)
            tokenizer_mock.save.assert_called_with('/tmp/my_tokenizer')
