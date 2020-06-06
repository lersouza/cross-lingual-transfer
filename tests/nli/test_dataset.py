import unittest

from unittest.mock import (
    Mock,
    mock_open,
    patch
)

from crosslangt.nli.dataset import (
    NLIDataset,
    NLIExample,
    load_mnli_dataset,
    parse_mnli_sample,
    load_assin_dataset
)


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
        dummy_mnli_body   = f'{MNLI_LINE_SAMPLE}\n' * 20
        dummy_mnli_file   = f'{dummy_mnli_header}\n{dummy_mnli_body}'

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


if __name__ == '__main__':
    unittest.main()


