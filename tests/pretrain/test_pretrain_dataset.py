import unittest
import random
import string
import torch

from unittest.mock import MagicMock, mock_open, patch
from torch import dtype
from torch.utils import data

from transformers import BertTokenizer
from crosslangt.pretrain.dataset import IndexEntry, LexicalTrainDataset
from crosslangt.pretrain.dataprep import create_from_document

RANDOM_FUNC = 'crosslangt.pretrain.dataprep.random'
DATASET_OPEN_FUNC = 'crosslangt.pretrain.dataset.open'
DATASET_PROCESS_ENTRY_FUNC = '_LexicalTrainDataset__process_entry'

BERT_TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased')


def get_mock_open_index():
    index_file = """location_a\t10
        location_b\t12
        location_c\t5"""

    return mock_open(read_data=index_file)


def load_location_mock_a(location):
    if location == 'location_a':
        return [{
            'input_ids': list(range(1, 10)),
            'token_type_ids': [0] * 9,
            'is_next': True
        }, {
            'input_ids': list(range(1, 10)),
            'token_type_ids': [0] * 9,
            'is_next': False
        }]
    else:
        return []


class GenerateDatasetTestCase(unittest.TestCase):
    def test_simple_document(self):
        # Generate a single document with 2 sentences of same size
        documents = self.generate_documents((5, 5), (2, 2), 1)

        # There should be only one training instance
        # We force it to never try a random sentence
        with patch(RANDOM_FUNC, return_value=0.9):
            instances = create_from_document(0, documents[0], documents, 10,
                                             BERT_TOKENIZER)

        self.assertEqual(
            len(instances), 1, 'There should be only one instance. '
            f'Documents={documents}. Instances={instances}')

        self.assertEqual(len(instances[0]['input_ids']), 10,
                         'The size of the example must be 10.')

        self.assertTrue(instances[0]['is_next'], 'No random sentences.')

        sent_a, sent_b = self.split_sentences(instances[0])

        self.assertGreater(len(sent_a), 0)
        self.assertGreater(len(sent_b), 0)

    def test_random_sentences(self):
        # Generating 2 documents for random next sentence test
        documents = self.generate_documents((5, 5), (2, 2), 2)

        with patch(RANDOM_FUNC, return_value=0.4):
            instances = create_from_document(0, documents[0], documents, 10,
                                             BERT_TOKENIZER)

        sizes = [len(i['input_ids']) for i in instances]
        classifications = [i['is_next'] for i in instances]

        self.assertEqual(len(instances), 2, 'There must be 2 examples')
        self.assertListEqual(sizes, [10, 10],
                             'All examples must be the same size')
        self.assertListEqual(classifications, [False, False],
                             'All examples must contain false next sentences')

    def test_short_documents(self):
        # We'll have 2 documents, each with 1 sentence
        # Every sentence will be of length 5.
        documents = self.generate_documents((5, 5), (1, 1), 2)

        # Forcing the algorithm to not go for a random sentence
        # based on probability
        with patch(RANDOM_FUNC, return_value=0.8):
            instances = create_from_document(0, documents[0], documents, 20,
                                             BERT_TOKENIZER)

        self.assertEqual(len(instances), 1, 'There must be 1 example')
        self.assertLess(
            len(instances[0]['input_ids']), 20,
            'The should be less than max_length since '
            'it does not have enough tokens.')
        self.assertFalse(instances[0]['is_next'],
                         'Should use a random sentence'
                         'for more data.')

    def split_sentences(self, training_instace):
        sentences = training_instace['input_ids']
        sentences = sentences[1:len(sentences) - 1]  # Trim CLS and last SEP

        sentence_sep = sentences.index(BERT_TOKENIZER.sep_token_id)

        return sentences[:sentence_sep], sentences[(sentence_sep + 1):]

    def generate_documents(self, sentence_min_max, doc_min_max, num_of_docs):
        documents = []

        for i in range(num_of_docs):
            num_of_sentences = random.randint(*doc_min_max)
            documents.append([])

            for s in range(num_of_sentences):
                sentence_size = random.randint(*sentence_min_max)
                documents[i].append([])

                for _ in range(sentence_size):
                    documents[i][s].append(''.join(
                        random.choices(string.ascii_letters,
                                       k=random.randint(1, 5))))

        return documents


@patch('crosslangt.pretrain.dataset.os.path.exists', return_value=True)
class LexicalTrainDatasetTestCase(unittest.TestCase):
    @patch('crosslangt.pretrain.dataset.torch.load', return_value=[])
    def test_index_creation(self, mock_load, exists_mock):
        expected_index = [
            IndexEntry(0, 9, 'location_a'),
            IndexEntry(10, 21, 'location_b'),
            IndexEntry(22, 26, 'location_c'),
        ]

        with patch(DATASET_OPEN_FUNC, get_mock_open_index()):
            dataset = LexicalTrainDataset('/some/index', BERT_TOKENIZER)

            self.assertEqual(len(dataset.index), 3)
            self.assertListEqual(dataset.index, expected_index)
            self.assertListEqual(dataset.examples, [])

    @patch('crosslangt.pretrain.dataset.torch.load', new=load_location_mock_a)
    def test_dataset_len(self, exists_mock):
        with patch(DATASET_OPEN_FUNC, get_mock_open_index()):
            dataset = LexicalTrainDataset('/some/index', BERT_TOKENIZER)

            self.assertEqual(len(dataset), 2)

    @patch('crosslangt.pretrain.dataset.torch.load', new=load_location_mock_a)
    def test_dataset_limit(self, exists_mock):
        with patch(DATASET_OPEN_FUNC, get_mock_open_index()):
            dataset = LexicalTrainDataset('/some/index',
                                          BERT_TOKENIZER,
                                          max_examples=1)

            # Check loaded examples
            self.assertEqual(len(dataset), 1)

            # Check total examples as per index
            self.assertEqual(dataset.total_examples, 27)

    @patch('crosslangt.pretrain.dataset.torch.load', new=load_location_mock_a)
    def test_dataset_getitem(self, exists_mock):
        expected_input_ids = torch.arange(1, 10, dtype=torch.long)
        expected_token_type_ids = torch.tensor([0] * 9, dtype=torch.long)
        expected_attention_mask = torch.ones(9)
        expected_is_next = torch.tensor([1], dtype=torch.long)

        with patch(DATASET_OPEN_FUNC, get_mock_open_index()):
            dataset = LexicalTrainDataset('/some/index', BERT_TOKENIZER)
            example = dataset[0]

            self.assertTrue(
                torch.all(torch.eq(example['input_ids'], expected_input_ids)))
            self.assertTrue(
                torch.all(
                    torch.eq(example['token_type_ids'],
                             expected_token_type_ids)))
            self.assertTrue(
                torch.all(
                    torch.eq(example['attention_mask'],
                             expected_attention_mask)))
            self.assertTrue(
                torch.all(
                    torch.eq(example['next_sentence_label'],
                             expected_is_next)))

    @patch('crosslangt.pretrain.dataset.torch.load', new=load_location_mock_a)
    def test_collate_batch(self, exists_mock):
        with patch(DATASET_OPEN_FUNC, get_mock_open_index()):
            dataset = LexicalTrainDataset('/some/index', BERT_TOKENIZER)
            examples = [dataset[i] for i in range(len(dataset))]

            batch = dataset.collate_batch(examples)

            self.assertIn('input_ids', batch)
            self.assertIn('token_type_ids', batch)
            self.assertIn('attention_mask', batch)
            self.assertIn('next_sentence_label', batch)
            self.assertIn('mlm_labels', batch)

            self.assertTupleEqual(batch['input_ids'].shape, (2, 9))
            self.assertTupleEqual(batch['token_type_ids'].shape, (2, 9))
            self.assertTupleEqual(batch['attention_mask'].shape, (2, 9))
            self.assertTupleEqual(batch['next_sentence_label'].shape, (2, ))
            self.assertTupleEqual(batch['mlm_labels'].shape, (2, 9))
