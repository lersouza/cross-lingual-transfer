import unittest
import random
import string


from unittest.mock import MagicMock, mock_open, patch
from crosslangt.pretrain.dataset import IndexEntry, LexicalTrainDataset
from crosslangt.pretrain.dataprep import create_from_document


RANDOM_FUNC = 'crosslangt.pretrain.dataprep.random'
DATASET_OPEN_FUNC = 'crosslangt.pretrain.dataset.open'
DATASET_PROCESS_ENTRY_FUNC = '_LexicalTrainDataset__process_entry'


class GenerateDatasetTestCase(unittest.TestCase):

    def test_simple_document(self):
        # Generate a single document with 2 sentences of same size
        documents = self.generate_documents(
            (5, 5), (2, 2), 1)

        # There should be only one training instance
        # We force it to never try a random sentence
        with patch(RANDOM_FUNC, return_value=0.9):
            instances = create_from_document(0, documents[0], documents, 10)

        self.assertEqual(len(instances), 1,
                         'There should be only one instance. '
                         f'Documents={documents}. Instances={instances}')

        self.assertEqual(len(instances[0].sentence_pair), 10,
                         'The size of the example must be 10.')

        self.assertTrue(instances[0].is_next, 'No random sentences.')

        sent_a, sent_b = self.split_sentences(instances[0])

        self.assertGreater(len(sent_a), 0)
        self.assertGreater(len(sent_b), 0)

    def test_random_sentences(self):
        # Generating 2 documents for random next sentence test
        documents = self.generate_documents(
            (5, 5), (2, 2), 2)

        with patch(RANDOM_FUNC, return_value=0.4):
            instances = create_from_document(0, documents[0], documents, 10)

        sizes = [len(i.sentence_pair) for i in instances]
        classifications = [i.is_next for i in instances]

        self.assertEqual(len(instances), 2, 'There must be 2 examples')
        self.assertListEqual(sizes, [10, 10],
                             'All examples must be the same size')
        self.assertListEqual(classifications, [False, False],
                             'All examples must contain false next sentences')

    def test_short_documents(self):
        # We'll have 2 documents, each with 1 sentence
        # Every sentence will be of length 5.
        documents = self.generate_documents(
            (5, 5), (1, 1), 2)

        # Forcing the algorithm to not go for a random sentence
        # based on probability
        with patch(RANDOM_FUNC, return_value=0.8):
            instances = create_from_document(0, documents[0], documents, 20)

        self.assertEqual(len(instances), 1, 'There must be 1 example')
        self.assertLess(len(instances[0].sentence_pair), 20,
                        'The should be less than max_length since '
                        'it does not have enough tokens.')
        self.assertFalse(instances[0].is_next, 'Should use a random sentence'
                                               'for more data.')

    def split_sentences(self, training_instace):
        sentences = training_instace.sentence_pair
        sentences = sentences[1:len(sentences) - 1]  # Trim CLS and last SEP

        sentence_sep = sentences.index('[SEP]')

        return sentences[:sentence_sep], sentences[(sentence_sep + 1):]

    def generate_documents(self, sentence_min_max, doc_min_max,
                           num_of_docs):
        documents = []

        for i in range(num_of_docs):
            num_of_sentences = random.randint(*doc_min_max)
            documents.append([])

            for s in range(num_of_sentences):
                sentence_size = random.randint(*sentence_min_max)
                documents[i].append([])

                for _ in range(sentence_size):
                    documents[i][s].append(
                        ''.join(
                            random.choices(
                                string.ascii_letters,
                                k=random.randint(1, 5))))

        return documents


class LexicalTrainDatasetTestCase(unittest.TestCase):

    @patch('crosslangt.pretrain.dataset.os.path.exists', return_value=True)
    def test_index_creation(self, exists_mock):
        index_file = """location_a\t10
        location_b\t12
        location_c\t5"""

        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        expected_index = [
            IndexEntry(0, 9, 'location_a'),
            IndexEntry(10, 21, 'location_b'),
            IndexEntry(22, 26, 'location_c'),
        ]

        with patch(DATASET_OPEN_FUNC, mock_open(read_data=index_file)):
            dataset = LexicalTrainDataset('/some/index', tokenizer)

            self.assertEqual(len(dataset.index), 3)
            self.assertListEqual(dataset.index, expected_index)

    @patch('crosslangt.pretrain.dataset.os.path.exists', return_value=True)
    def test_multiple_files(self, exists_mock):
        index_file = """location_a\t10
        location_b\t12
        location_c\t5"""

        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0

        with patch(DATASET_OPEN_FUNC, mock_open(read_data=index_file)):
            dataset = LexicalTrainDataset('/some/index', tokenizer)

            with patch.object(dataset, DATASET_PROCESS_ENTRY_FUNC) as pemock:
                pemock.side_effect = [
                    (x for x in range(10)),
                    (x for x in range(12)),
                    (x for x in range(5)),
                ]

                expected = list(range(10)) + list(range(12)) + list(range(5))
                actual = [el for el in dataset]

                self.assertListEqual(actual, expected)
