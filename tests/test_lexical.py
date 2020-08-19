import torch

from torch.nn import Embedding
from transformers import BertModel, BertTokenizer

from crosslangt.lexical import (SlicedEmbedding, setup_lexical_for_testing,
                                setup_lexical_for_training)

from unittest.case import TestCase
from unittest.mock import patch


class SimpleNetwork(torch.nn.Module):
    def __init__(self, embeddings: torch.nn.Embedding) -> None:
        super(SimpleNetwork, self).__init__()

        self.embeddings = embeddings
        self.linear = torch.nn.Linear(embeddings.embedding_dim, 1)

    def forward(self, batch):
        batch = self.embeddings(batch)
        batch = self.linear(batch)

        return batch


class SlicedEmbeddingsTest(TestCase):

    def test_lookup_embeddings(self):
        original_embedding = torch.nn.Embedding(10, 2)
        slieced_embedding = SlicedEmbedding.slice(
            original_embedding, 5, True, False)

        batch_for_original = torch.tensor([[0, 1], [8, 9]])
        batch_for_slieced = torch.tensor([[0, 1], [8, 9]])

        lookup_original = original_embedding(batch_for_original)
        lookup_sliced = slieced_embedding(batch_for_slieced)

        self.assertTrue(torch.all(lookup_original == lookup_sliced).item())

    def test_freeze_initial_positions(self):
        """
        We emulate a simple training loop to check whether embeddings
        are being updated correctly.
        """
        original_embedding = torch.nn.Embedding(10, 2)
        slieced_embedding = SlicedEmbedding.slice(
            original_embedding, 5, True, False)

        # We clone the original weigths, since they are updated
        original_values = original_embedding.weight.clone()

        data = torch.tensor([[[0, 1, 5], [2, 3, 6]], [[0, 1, 9], [4, 5, 7]]])
        labels = torch.tensor([[0], [1]])  # Always fixed

        model = SimpleNetwork(slieced_embedding)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        for i in range(2):
            model.train()
            optimizer.zero_grad()

            for i, batch in enumerate(data):
                outputs = model(batch)
                loss = torch.nn.functional.cross_entropy(outputs, labels)

                loss.backward()
                optimizer.step()

        freezed_original = original_values[:5]
        trainable_original = original_values[5:]

        freezed_actual = model.embeddings.first_embedding.weight
        trainable_actual = model.embeddings.second_embedding.weight

        self.assertTrue(torch.all(freezed_original == freezed_actual).item())
        self.assertFalse(
            torch.all(trainable_original == trainable_actual).item())

    def test_freeze_final_positions(self):
        """
        We emulate a simple training loop to check whether embeddings
        are being updated correctly.
        """
        original_embedding = torch.nn.Embedding(10, 2)
        slieced_embedding = SlicedEmbedding.slice(
            original_embedding, 5, False, True)

        # We clone the original weigths, since they are updated
        original_values = original_embedding.weight.clone()

        data = torch.tensor([[[0, 1, 5], [2, 3, 6]], [[0, 1, 9], [4, 5, 7]]])
        labels = torch.tensor([[0], [1]])  # Always fixed

        model = SimpleNetwork(slieced_embedding)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        for i in range(2):
            model.train()
            optimizer.zero_grad()

            for i, batch in enumerate(data):
                outputs = model(batch)
                loss = torch.nn.functional.cross_entropy(outputs, labels)

                loss.backward()
                optimizer.step()

        trainable_original = original_values[:5]
        freezed_original = original_values[5:]

        trainable_actual = model.embeddings.first_embedding.weight
        freezed_actual = model.embeddings.second_embedding.weight

        self.assertTrue(torch.all(freezed_original == freezed_actual).item())
        self.assertFalse(
            torch.all(trainable_original == trainable_actual).item())

    def test_slice(self):
        original = Embedding(10, 10)
        cut = 5

        sliced = SlicedEmbedding.slice(original, 5, True, False)

        self.assertTrue(type(sliced) is SlicedEmbedding)

        first = sliced.first_embedding
        second = sliced.second_embedding

        self.assertFalse(first.weight.requires_grad)
        self.assertTrue(second.weight.requires_grad)

        self.assertTrue(
            torch.all(original.weight[:cut] == first.weight).item())

        self.assertTrue(
            torch.all(original.weight[cut:] == second.weight).item())

    def test_slice_again(self):
        original = Embedding(10, 10)
        already_sliced = SlicedEmbedding.slice(original, 5, True, False)
        slice_it_again = SlicedEmbedding.slice(already_sliced, 5, True, False)

        self.assertTrue(type(slice_it_again) is SlicedEmbedding)
        self.assertEqual(already_sliced, slice_it_again)

    def test_slice_again_different_cut(self):
        original = Embedding(10, 10)
        already_sliced = SlicedEmbedding.slice(original, 5, True, False)

        with self.assertRaises(NotImplementedError) as context:
            SlicedEmbedding.slice(already_sliced, 6, True, False)


class SetupLexicalForTrainingTest(TestCase):
    def test_setup_none(self):
        model = BertModel.from_pretrained('bert-base-cased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        setup_lexical_for_training('none', model, tokenizer)

        self.assertTrue(model.get_input_embeddings().weight.requires_grad)

    def test_setup_freeze_special(self):
        model = BertModel.from_pretrained('bert-base-cased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        setup_lexical_for_training('freeze-special', model, tokenizer)

        lexical = model.get_input_embeddings()
        lexical_type = type(lexical)

        self.assertTrue(lexical_type is SlicedEmbedding)
        self.assertFalse(lexical.first_embedding.weight.requires_grad)
        self.assertTrue(lexical.second_embedding.weight.requires_grad)

    def test_setup_freeze_nonspecial(self):
        model = BertModel.from_pretrained('bert-base-cased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        setup_lexical_for_training('freeze-nonspecial', model, tokenizer)

        lexical = model.get_input_embeddings()
        lexical_type = type(lexical)

        self.assertTrue(lexical_type is SlicedEmbedding)
        self.assertTrue(lexical.first_embedding.weight.requires_grad)
        self.assertFalse(lexical.second_embedding.weight.requires_grad)

    def test_setup_freeze_all(self):
        model = BertModel.from_pretrained('bert-base-cased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        setup_lexical_for_training('freeze-all', model, tokenizer)

        lexical = model.get_input_embeddings()
        lexical_type = type(lexical)

        self.assertTrue(lexical_type is Embedding)
        self.assertFalse(lexical.weight.requires_grad)


class SetupLexicalForTestingTest(TestCase):
    def test_setup_original(self):
        model = BertModel.from_pretrained('bert-base-cased')
        model_embedding = model.get_input_embeddings()
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        setup_lexical_for_testing('original', model, tokenizer, None)

        new_lexical = model.get_input_embeddings()
        new_lexical_type = type(new_lexical)

        self.assertTrue(new_lexical_type is Embedding)
        self.assertEqual(model_embedding, new_lexical)

    def test_setup_target_lexical(self):
        model = BertModel.from_pretrained('bert-base-cased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        target_emb = torch.nn.Embedding(10, 768)

        with patch('crosslangt.lexical.torch.load',
                   return_value={'weight': target_emb.weight}):

            setup_lexical_for_testing('target-lexical', model, tokenizer,
                                      'some_path')

        new_lexical = model.get_input_embeddings()
        new_lexical_type = type(new_lexical)

        self.assertTrue(new_lexical_type is Embedding)
        self.assertTrue(
            torch.all(new_lexical.weight == target_emb.weight).item())

    @patch('crosslangt.lexical.PreTrainedTokenizer.all_special_ids', [0])
    def test_setup_target_lexical_special_original(self):
        model = BertModel.from_pretrained('bert-base-cased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        target = torch.nn.Embedding(10, 768)
        original = model.get_input_embeddings()

        with patch('crosslangt.lexical.torch.load',
                   return_value={'weight': target.weight}):

            setup_lexical_for_testing('target-lexical-original-special', model,
                                      tokenizer, 'some_path')

        new_lexical = model.get_input_embeddings()
        new_lexical_type = type(new_lexical)

        self.assertTrue(new_lexical_type is SlicedEmbedding)

        fe = new_lexical.first_embedding
        se = new_lexical.second_embedding

        self.assertTrue(torch.all(fe.weight == original.weight[:1]).item())
        self.assertTrue(torch.all(se.weight == target.weight[1:]).item())
