import torch
import unittest

from unittest.mock import (
    Mock,
    mock_open,
    patch
)

from crosslangt.nli.dataset import (
    DEFAULT_LABELS,
    NLIDataset,
    NLIExample,
    load_mnli_dataset,
    parse_mnli_sample,
    load_assin_dataset
)

from transformers import BertTokenizer


MNLI_LINE_SAMPLE = '9814\t76653\t76653c\ttelephone\t\t\t\t\ti\'ll listen  and agree with what i think sounds right\tI wont even bother listening.\tcontradiction\tcontradiction\tcontradiction\tcontradiction \tcontradiction \tcontradiction'


class DatasetsTestCase(unittest.TestCase):

    @patch('os.path.exists', lambda file: True)
    def test_load_mnli_dataset(self):
        # The MNLI file contains the following format:
        # - It is a Tab Separated Values file (tsv)
        # - It contains 15 columns
        # - The first line is the header line, with column names
        # - Next lines are all dataset examples
        dummy_mnli_header = '\t'.join(['dummyheader'] * 15)
        dummy_mnli_body = f'{MNLI_LINE_SAMPLE}\n' * 20
        dummy_mnli_file = f'{dummy_mnli_header}\n{dummy_mnli_body}'

        bert_tokenizer = Mock()

        open_mock = mock_open(read_data=dummy_mnli_file)
        with patch('builtins.open', open_mock):
            dataset = load_mnli_dataset('any_root', 'any_file',
                                        bert_tokenizer, 256)

        self.assertEqual(len(dataset), 20)
        self.assertEqual(dataset.tokenizer, bert_tokenizer)
        self.assertEqual(dataset.max_seq_length, 256)

    def test_parse_valid_mnli_sample(self):
        example = parse_mnli_sample(0, MNLI_LINE_SAMPLE)

        self.assertEqual(example.pair_id, '76653c')
        self.assertEqual(example.original_index, 0)
        self.assertTrue(
            example.sentence_a,
            'i\'ll listen  and agree with what i think sounds right')
        self.assertEqual(
            example.sentence_b,
            'I wont even bother listening.')
        self.assertEqual(example.label, 'contradiction')

    def test_dataset_sample_contract(self):
        dataset = self.__build_demo_dataset()
        dataset_sample = dataset[0]
        dataset_sample_keys = dataset_sample.keys()

        self.assertIn('pair_id', dataset_sample_keys)
        self.assertIn('input_ids', dataset_sample_keys)
        self.assertIn('attention_mask', dataset_sample_keys)
        self.assertIn('token_type_ids', dataset_sample_keys)
        self.assertIn('label', dataset_sample_keys)

    def test_dataset_sample_tokenization(self):
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        dataset = self.__build_demo_dataset(tokenizer=bert_tokenizer,
                                            max_seq_length=15)
        sample = dataset[0]
        original_example = dataset.examples[0]

        encoded = bert_tokenizer.encode_plus(
            original_example.sentence_a,
            original_example.sentence_b,
            max_length=15,
            pad_to_max_length=True,
            return_tensors='pt')

        all = torch.all
        is_true = self.assertTrue

        is_true(sample['pair_id'] == original_example.pair_id)
        is_true(all(sample['input_ids'] == encoded['input_ids']))
        is_true(all(sample['attention_mask'] == encoded['attention_mask']))
        is_true(all(sample['token_type_ids'] == encoded['token_type_ids']))
        is_true(sample['label'] == DEFAULT_LABELS.index('neutral'))

    def __build_demo_dataset(self, tokenizer=None, max_seq_length=15):
        examples = [
            NLIExample('1', 0, 'some sentence', 'another sentence', 'neutral'),
            NLIExample('2', 1, 'afirmation', 'contradict', 'contraditction'),
            NLIExample('3', 2, 'afirmation 2', 'entailment', 'entailment')
        ]

        # Perhaps I should mock the tokenizer?
        tokenizer = tokenizer or BertTokenizer.from_pretrained(
            'bert-base-cased')
        dataset = NLIDataset(examples, tokenizer,
                             max_seq_length=max_seq_length)

        return dataset


if __name__ == '__main__':
    unittest.main()
