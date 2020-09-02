import torch
from torch import tensor

from crosslangt.nli.modeling_nli import NLIFinetuneModel
from pytorch_lightning.metrics import Accuracy
from unittest import TestCase
from unittest.mock import MagicMock, call, patch

BASE_NAME = 'crosslangt.nli.modeling_nli'


@patch(BASE_NAME + '.AutoTokenizer.from_pretrained')
@patch(BASE_NAME + '.AutoModelForSequenceClassification.from_pretrained')
@patch(BASE_NAME + '.AutoConfig.from_pretrained')
class NLIModelTestCase(TestCase):
    def test_model_load(self, config_mock, model_mock, tok_mock):
        config, pretrained, tokenizer = self.__setup_mocks(
            config_mock, model_mock, tok_mock)

        model = NLIFinetuneModel(pretrained_model='bert-base-cased',
                                 num_classes=3,
                                 train_lexical_strategy='none',
                                 train_dataset='mnli',
                                 eval_dataset='mnli',
                                 data_dir='/some/data/dir',
                                 batch_size=32,
                                 max_seq_length=128)

        config_mock.assert_called_with('bert-base-cased', num_labels=3)
        model_mock.assert_called_with('bert-base-cased', config=config)
        tok_mock.assert_called_with('bert-base-cased')

        self.assertEqual(model.train_key, 'bert-base-cased.')
        self.assertEqual(model.hparams.pretrained_model, 'bert-base-cased')
        self.assertIsInstance(model.metric, Accuracy)

        self.assertEqual(model.model, pretrained)
        self.assertEqual(model.tokenizer, tokenizer)

    @patch(BASE_NAME + '.load_nli_dataset')
    @patch(BASE_NAME + '.setup_lexical_for_training')
    def test_setup(self, setup_mock, load_mock, config_mock, model_mock,
                   tok_mock):

        _, pretrained, tokenizer = self.__setup_mocks(config_mock, model_mock,
                                                      tok_mock)

        load_mock.return_value = 'some_data_set'

        model = NLIFinetuneModel(pretrained_model='bert-base-cased',
                                 num_classes=3,
                                 train_lexical_strategy='my-strategy',
                                 train_dataset='mnli',
                                 eval_dataset='mnli-for-eval',
                                 data_dir='/data',
                                 batch_size=32,
                                 max_seq_length=128)

        model.setup('fit')

        setup_mock.assert_called_with('my-strategy', pretrained, tokenizer)
        load_mock.assert_has_calls([
            call('/data', 'mnli', 'train', 128, 'bert-base-cased.'),
            call('/data', 'mnli-for-eval', 'eval', 128, 'bert-base-cased.')
        ])

        self.assertEqual(model.train_dataset, 'some_data_set')
        self.assertEqual(model.eval_dataset, 'some_data_set')

    def test_train_step(self, config_mock, model_mock, tok_mock):
        _, pretrained, tokenizer = self.__setup_mocks(config_mock, model_mock,
                                                      tok_mock)

        model = NLIFinetuneModel(pretrained_model='bert-base-cased',
                                 num_classes=3,
                                 train_lexical_strategy='my-strategy',
                                 train_dataset='mnli',
                                 eval_dataset='mnli-for-eval',
                                 data_dir='/data',
                                 batch_size=32,
                                 max_seq_length=128)

        batch = self.__get_batch()

        with patch.object(model, 'forward', self.__mock_forward):
            results = model.training_step(batch, 1)

            self.assertTrue('loss' in results)
            self.assertTrue('log' in results)

            self.assertEqual(results['loss'], torch.tensor(0.3))
            self.assertEqual(results['log']['train_acc'], torch.tensor(0.5))

    def test_valid_step(self, config_mock, model_mock, tok_mock):
        _, pretrained, tokenizer = self.__setup_mocks(config_mock, model_mock,
                                                      tok_mock)

        model = NLIFinetuneModel(pretrained_model='bert-base-cased',
                                 num_classes=3,
                                 train_lexical_strategy='my-strategy',
                                 train_dataset='mnli',
                                 eval_dataset='mnli-for-eval',
                                 data_dir='/data',
                                 batch_size=32,
                                 max_seq_length=128)

        batch = self.__get_batch()

        with patch.object(model, 'forward', self.__mock_forward):
            results = model.validation_step(batch, 1)

            self.assertTrue('val_loss' in results)
            self.assertTrue('log' in results)

            self.assertEqual(results['val_loss'], torch.tensor(0.3))
            self.assertEqual(results['log']['val_acc'], torch.tensor(0.5))

    def test_test_step(self, config_mock, model_mock, tok_mock):
        _, pretrained, tokenizer = self.__setup_mocks(config_mock, model_mock,
                                                      tok_mock)

        model = NLIFinetuneModel(pretrained_model='bert-base-cased',
                                 num_classes=3,
                                 train_lexical_strategy='my-strategy',
                                 train_dataset='mnli',
                                 eval_dataset='mnli-for-eval',
                                 data_dir='/data',
                                 batch_size=32,
                                 max_seq_length=128)

        batch = self.__get_batch()

        with patch.object(model, 'forward', self.__mock_forward):
            results = model.test_step(batch, 1)

            self.assertTrue('test_loss' in results)
            self.assertTrue('log' in results)

            self.assertEqual(results['test_loss'], torch.tensor(0.3))
            self.assertEqual(results['log']['test_acc'], torch.tensor(0.5))

    def test_valid_epoch_end(self, config_mock, model_mock, tok_mock):
        _, pretrained, tokenizer = self.__setup_mocks(config_mock, model_mock,
                                                      tok_mock)

        model = NLIFinetuneModel(pretrained_model='bert-base-cased',
                                 num_classes=3,
                                 train_lexical_strategy='my-strategy',
                                 train_dataset='mnli',
                                 eval_dataset='mnli-for-eval',
                                 data_dir='/data',
                                 batch_size=32,
                                 max_seq_length=128)

        outputs = [
            {
                'val_loss': torch.tensor(1.),
                'val_acc': torch.tensor(0.5)
            },
            {
                'val_loss': torch.tensor(2.),
                'val_acc': torch.tensor(0.7)
            },
            {
                'val_loss': torch.tensor(3.),
                'val_acc': torch.tensor(0.4)
            },
            {
                'val_loss': torch.tensor(4.),
                'val_acc': torch.tensor(0.4)
            },
        ]

        results = model.validation_epoch_end(outputs)

        self.assertTrue('val_avg_loss' in results)
        self.assertTrue('val_avg_accuracy' in results)
        self.assertTrue('log' in results)

        self.assertEqual(results['val_avg_loss'], torch.tensor(2.5))
        self.assertEqual(results['val_avg_accuracy'], torch.tensor(0.5))

    def test_test_epoch_end(self, config_mock, model_mock, tok_mock):
        _, pretrained, tokenizer = self.__setup_mocks(config_mock, model_mock,
                                                      tok_mock)

        model = NLIFinetuneModel(pretrained_model='bert-base-cased',
                                 num_classes=3,
                                 train_lexical_strategy='my-strategy',
                                 train_dataset='mnli',
                                 eval_dataset='mnli-for-eval',
                                 data_dir='/data',
                                 batch_size=32,
                                 max_seq_length=128)

        outputs = [
            {
                'test_loss': torch.tensor(1.),
                'test_acc': torch.tensor(0.5)
            },
            {
                'test_loss': torch.tensor(2.),
                'test_acc': torch.tensor(0.7)
            },
            {
                'test_loss': torch.tensor(3.),
                'test_acc': torch.tensor(0.4)
            },
            {
                'test_loss': torch.tensor(4.),
                'test_acc': torch.tensor(0.4)
            },
        ]

        results = model.test_epoch_end(outputs)

        self.assertTrue('test_avg_loss' in results)
        self.assertTrue('test_avg_accuracy' in results)
        self.assertTrue('log' in results)

        self.assertEqual(results['test_avg_loss'], torch.tensor(2.5))
        self.assertEqual(results['test_avg_accuracy'], torch.tensor(0.5))

    def test_batch_logging(self, config_mock, model_mock, tok_mock):
        _, pretrained, tokenizer = self.__setup_mocks(config_mock, model_mock,
                                                      tok_mock)

        tokenizer.decode.return_value = 'a a a [SEP] b b'
        tokenizer.sep_token = '[SEP]'

        logger = MagicMock()
        logger.experiment = MagicMock()
        logger.experiment.add_text = MagicMock()

        model = NLIFinetuneModel(pretrained_model='bert-base-cased',
                                 num_classes=3,
                                 train_lexical_strategy='my-strategy',
                                 train_dataset='mnli',
                                 eval_dataset='mnli-for-eval',
                                 data_dir='/data',
                                 batch_size=32,
                                 max_seq_length=128)

        model.logger = logger

        batch = self.__get_batch()

        with patch.object(model, 'forward', self.__mock_forward):
            model.training_step(batch, 1)
            logger.experiment.add_text.assert_not_called()

            model.validation_step(batch, 1)
            logger.experiment.add_text.assert_has_calls([
                call.__bool__(),
                call('a a a\tb b\t0\t0'),
                call('a a a\tb b\t1\t0')
            ])

            model.test_step(batch, 1)
            logger.experiment.add_text.assert_has_calls([
                call.__bool__(),
                call('a a a\tb b\t0\t0'),
                call('a a a\tb b\t1\t0'),
                call.__bool__(),
                call('a a a\tb b\t0\t0'),
                call('a a a\tb b\t1\t0')
            ])

    def __get_batch(self):
        return {
            'input_ids':
            torch.tensor([[-100, -100, -100, -100], [100, 100, 100, 100]],
                         dtype=torch.long),
            'attention_mask':
            torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.long),
            'token_type_ids':
            torch.tensor([[2, 2, 2, 2], [2, 2, 2, 2]], dtype=torch.long),
            'label':
            torch.tensor([0, 1], dtype=torch.long)
        }

    def __mock_forward(self, **inputs):
        """
        Mock function for the model's forward.
        If labels are provided, return a fixed loss of 0.3.
        Always return logits such that the class 0 will be the resulting class.
        """
        return_object = (torch.tensor([[0.7, 0.6, 0.5, 0.4],
                                       [0.7, 0.6, 0.5,
                                        0.4]]), torch.tensor([]),
                         torch.tensor([]))

        if 'labels' in inputs:
            return_object = (torch.tensor([0.3]), ) + return_object

        return return_object

    def __setup_mocks(self, config_mock, model_mock, tokenizer_mock):
        config = MagicMock()
        pretrained_model = MagicMock()
        tokenizer = MagicMock()

        config_mock.return_value = config
        model_mock.return_value = pretrained_model
        tokenizer_mock.return_value = tokenizer

        return config, pretrained_model, tokenizer
