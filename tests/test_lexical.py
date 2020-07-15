import torch

from torch.nn import Embedding
from transformers import BertModel, BertConfig, BertTokenizer

from crosslangt.lexical import SlicedEmbeddings, setup_lexical

from unittest.case import TestCase


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
        slieced_embedding = SlicedEmbeddings(
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
        slieced_embedding = SlicedEmbeddings(
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
        slieced_embedding = SlicedEmbeddings(
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


class SetupLexicalTest(TestCase):
    def test_setup_none(self):
        model = BertModel.from_pretrained('bert-base-cased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        setup_lexical('none', model, tokenizer)

        self.assertTrue(model.get_input_embeddings().weight.requires_grad)

    def test_setup_freeze_special(self):
        model = BertModel.from_pretrained('bert-base-cased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        setup_lexical('freeze-special', model, tokenizer)

        lexical = model.get_input_embeddings()
        lexical_type = type(lexical)

        self.assertTrue(lexical_type is SlicedEmbeddings)
        self.assertFalse(lexical.first_embedding.weight.requires_grad)
        self.assertTrue(lexical.second_embedding.weight.requires_grad)

    def test_setup_freeze_nonspecial(self):
        model = BertModel.from_pretrained('bert-base-cased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        setup_lexical('freeze-nonspecial', model, tokenizer)

        lexical = model.get_input_embeddings()
        lexical_type = type(lexical)

        self.assertTrue(lexical_type is SlicedEmbeddings)
        self.assertTrue(lexical.first_embedding.weight.requires_grad)
        self.assertFalse(lexical.second_embedding.weight.requires_grad)

    def test_setup_freeze_all(self):
        model = BertModel.from_pretrained('bert-base-cased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

        setup_lexical('freeze-all', model, tokenizer)

        lexical = model.get_input_embeddings()
        lexical_type = type(lexical)

        self.assertTrue(lexical_type is Embedding)
        self.assertFalse(lexical.weight.requires_grad)
