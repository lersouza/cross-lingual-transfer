import unittest

from crosslangt.nli.dataset import (
    NLIDataset,
    NLIExample,
    load_mnli_dataset,
    parse_mnli_sample,
    load_assin_dataset
)


MNLI_LINE_SAMPLE = '9814\t76653\t76653c\ttelephone\t\t\t\t\ti\'ll listen  and agree with what i think sounds right\tI wont even bother listening.\tcontradiction\tcontradiction\tcontradiction\tcontradiction \tcontradiction \tcontradiction'


class LoadDatasetsTestCase(unittest.TestCase):

    def test_parse_valid_mnli_sample(self):
        example = parse_mnli_sample(0, MNLI_LINE_SAMPLE)
        
        self.assertEqual(example.pairID, '76653c')
        self.assertEqual(example.originalIndex, 0)
        self.assertTrue(
            example.sentence_a,
            'i\'ll listen  and agree with what i think sounds right')
        self.assertEqual(
            example.sentence_b,
            'I wont even bother listening.')
        self.assertEqual(example.label, 'contradiction')


if __name__ == '__main__':
    unittest.main()


