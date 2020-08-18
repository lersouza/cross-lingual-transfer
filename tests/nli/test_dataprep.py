from unittest import TestCase
from unittest.mock import call, patch

from crosslangt.nli import (extract_features, get_features_file,
                            MnliNoContradictionProcessor)
from tests.mocking import get_tokenizer
from transformers.data.processors.utils import InputExample


class DataProcessorMock():
    def get_dev_examples(self, data_dir):
        return [
            InputExample('1', 'text+a+eval', 'text+b', 'entailment'),
            InputExample('2', 'text+c+eval', 'text+d', 'neutral'),
        ]

    def get_train_examples(self, data_dir):
        return [
            InputExample('3', 'text+a', 'text+b', 'neutral'),
            InputExample('4', 'text+c', 'text+d', 'entailment'),
            InputExample('5', 'text+e', 'text+f', 'entailment'),
        ]

    def get_labels(self):
        return ['neutral', 'entailment']


class NliDataPrepTestCase(TestCase):
    def test_feature_file_name(self):
        fkey = 'some_key'
        dataset = 'assin2'
        split = 'train'
        max_seq_length = 128

        expected = '/data/nli-some_key-assin2-train-128.dataset'
        actual = get_features_file('/data', dataset, split, max_seq_length,
                                   fkey)

        self.assertEqual(expected, actual)

    def test_extract_train_features(self):
        tokenizer = get_tokenizer()
        processor = DataProcessorMock()

        features = extract_features('/data/train.tsv', 'train', 64, tokenizer,
                                    processor)

        self.assertEqual(len(features), 3)

        self.assertEqual(features[0].label, 0)
        self.assertEqual(features[1].label, 1)
        self.assertEqual(features[2].label, 1)

        self.assertEqual(features[0].pairID, '3')
        self.assertEqual(features[1].pairID, '4')
        self.assertEqual(features[2].pairID, '5')

        tokenizer.encode_plus.assert_has_calls([
            call('text+a', 'text+b', max_length=64, pad_to_max_length=True),
            call('text+c', 'text+d', max_length=64, pad_to_max_length=True),
            call('text+e', 'text+f', max_length=64, pad_to_max_length=True)
        ])

    def test_extract_dev_features(self):
        tokenizer = get_tokenizer()
        processor = DataProcessorMock()

        features = extract_features('/data/train.tsv', 'eval', 64, tokenizer,
                                    processor)

        self.assertEqual(len(features), 2)

        self.assertEqual(features[0].label, 1)
        self.assertEqual(features[1].label, 0)

        self.assertEqual(features[0].pairID, '1')
        self.assertEqual(features[1].pairID, '2')

        tokenizer.encode_plus.assert_has_calls([
            call('text+a+eval',
                 'text+b',
                 max_length=64,
                 pad_to_max_length=True),
            call('text+c+eval',
                 'text+d',
                 max_length=64,
                 pad_to_max_length=True),
        ])


class NliProcessorsTestCase(TestCase):
    def test_mnli_train_no_contradiction(self):
        processor = MnliNoContradictionProcessor()
        base_name = 'crosslangt.nli.dataprep_nli.MnliProcessor' \
                    '.get_train_examples'
        examples = [
            InputExample('1', 'text a', 'text b', 'contradiction'),
            InputExample('2', 'text c', 'text e', 'entailment'),
            InputExample('3', 'text d', 'text f', 'neutral'),
        ]

        expected = examples[1:]  # All but the contradiction example

        with patch(base_name, return_value=examples):
            actual = processor.get_train_examples('/some/data/dir')
            self.assertListEqual(expected, list(actual))

    def test_mnli_dev_no_contradiction(self):
        processor = MnliNoContradictionProcessor()
        base_name = 'crosslangt.nli.dataprep_nli.MnliProcessor' \
                    '.get_dev_examples'
        examples = [
            InputExample('2', 'text c', 'text e', 'entailment'),
            InputExample('3', 'text d', 'text f', 'neutral'),
            InputExample('1', 'text a', 'text b', 'contradiction')
        ]

        expected = examples[:2]  # All but the contradiction example

        with patch(base_name, return_value=examples):
            actual = processor.get_dev_examples('/some/data/dir')
            self.assertListEqual(expected, list(actual))

    def test_mnli_no_contradiction_labels(self):
        processor = MnliNoContradictionProcessor()

        actual = processor.get_labels()
        expected = ['entailment', 'neutral']

        self.assertListEqual(expected, actual)

    def test_in_extract_features(self):
        processor = MnliNoContradictionProcessor()
        tokenizer = get_tokenizer()
        base_name = 'crosslangt.nli.dataprep_nli.MnliProcessor' \
                    '.get_dev_examples'
        examples = [
            InputExample('2', 'text c', 'text e', 'entailment'),
            InputExample('3', 'text d', 'text f', 'neutral'),
            InputExample('1', 'text a', 'text b', 'contradiction')
        ]

        with patch(base_name, return_value=examples):
            features = extract_features('/some/data/train.tsv', 'eval',
                                        64, tokenizer, processor)

            self.assertEqual(len(features), 2)

            self.assertEqual(features[0].label, 0)
            self.assertEqual(features[1].label, 1)

            self.assertEqual(features[0].pairID, '2')
            self.assertEqual(features[1].pairID, '3')
