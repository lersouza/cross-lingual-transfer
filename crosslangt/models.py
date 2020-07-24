import os
from pytorch_lightning.trainer.trainer import Trainer

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import Accuracy
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from transformers import (BertConfig, BertForQuestionAnswering,
                          BertForSequenceClassification, BertTokenizer)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits, squad_evaluate)
from transformers.data.processors.squad import SquadFeatures, SquadResult

from crosslangt.dataprep import (load_nli_dataset,
                                 load_question_answer_dataset,
                                 prepare_nli_dataset,
                                 prepare_question_answer_dataset)
from crosslangt.lexical import (setup_lexical_for_training,
                                setup_lexical_for_testing)


class QuestionAnsweringModel(LightningModule):
    """
    A model for Question-Answering. This model is suitable for training
    and finetuning with SQuAD 1.1 compatible datasets.
    """

    def __init__(self,
                 pretrained_model: str,
                 train_lexical_strategy: str,
                 test_lexical_strategy: str,
                 train_dataset: str,
                 test_dataset: str,
                 data_dir: str,
                 batch_size: int,
                 max_seq_length: int,
                 max_query_length: int,
                 doc_stride: int,
                 output_dir: str,
                 n_best_size: int = 20,
                 max_answer_length: int = 30,
                 test_lexical_path: str = None,
                 tokenizer_name: str = None,
                 test_tokenizer_name: str = None,
                 **kwargs) -> None:

        super(QuestionAnsweringModel, self).__init__()

        self.save_hyperparameters()

        self.train_tokenizer, self.test_tokenizer = self.__get_tokenizers()
        self.bert = BertForQuestionAnswering.from_pretrained(pretrained_model)

    def forward(self, input_ids, attention_mask, token_type_ids,
                start_positions=None, end_positions=None):

        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions
        )

        return outputs

    def configure_optimizers(self):
        return Adam(self.bert.parameters(), lr=2e-5)

    def training_step(self, batch, batch_idx):
        parameters = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'start_positions': batch[3],
            'end_positions': batch[4]
        }

        outputs = self(**parameters)
        loss = outputs[0]

        logs = {'train_loss': loss}

        return {'loss': loss, 'log': logs}

    def test_step(self, batch, batch_idx):
        parameters = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
        }
        feature_indices = batch[3]

        outputs = self(**parameters)
        start_scores, end_scores = outputs[:2]

        results = []

        for i, feature_index in enumerate(feature_indices):
            eval_feature = self.eval_features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            result = SquadResult(
                unique_id,
                start_scores[i].detach().cpu().tolist(),
                end_scores[i].detach().cpu().tolist()
            )

            results.append(result)

        return {'results': results}

    def test_epoch_end(self, outputs):
        all_results = []

        for output in outputs:
            all_results.extend(output['results'])

        output_prediction_file = os.path.join(
            self.hparams.output_dir,
            f'predictions_epoch{self.current_epoch}.json')

        output_nbest_file = os.path.join(
            self.hparams.output_dir,
            f'nbest_predictions_epoch{self.current_epoch}.json')

        examples, features = self.__retrieve_eval_feature_set(all_results)

        predictions = compute_predictions_logits(
            examples,
            features,
            all_results,
            self.hparams.n_best_size,
            self.hparams.max_answer_length,
            False,
            output_prediction_file,
            output_nbest_file,
            None,
            False,
            False,
            0.0,
            self.test_tokenizer,
        )

        results = squad_evaluate(examples, predictions)

        return {
            'exact': torch.tensor(results['exact']),
            'f1': torch.tensor(results['f1']),
            'total': torch.tensor(results['total'])
        }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.hparams.batch_size,
            num_workers=3
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.eval_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            num_workers=3
        )

    def prepare_data(self):
        prepare_question_answer_dataset(
            self.hparams.dataset,
            'train',
            self.hparams.data_dir,
            self.train_tokenizer,
            self.hparams.max_seq_length,
            self.hparams.doc_stride,
            self.hparams.max_query_length
        )

        prepare_question_answer_dataset(
            self.hparams.dataset,
            'eval',
            self.hparams.data_dir,
            self.test_tokenizer,
            self.hparams.max_seq_length,
            self.hparams.doc_stride,
            self.hparams.max_query_length
        )

    def setup(self, stage: str):
        if stage == 'fit':
            setup_lexical_for_training(
                self.hparams.train_lexical_strategy,
                self.bert,
                self.train_tokenizer)

            train_data = load_question_answer_dataset(
                self.hparams.dataset,
                'train',
                self.hparams.data_dir,
                self.hparams.max_seq_length
            )
            self.train_dataset = train_data['dataset']

        elif stage == 'test':
            setup_lexical_for_testing(
                self.hparams.test_lexical_strategy,
                self.bert,
                self.test_tokenizer,
                self.hparams.test_lexical_path
            )

            eval_data = load_question_answer_dataset(
                self.hparams.dataset,
                'eval',
                self.hparams.data_dir,
                self.hparams.max_seq_length
            )

            self.eval_dataset = eval_data['dataset']
            self.eval_examples = eval_data['examples']
            self.eval_features = eval_data['features']
            self.eval_features_index = {
                f.unique_id: f for f in self.eval_features}

        os.makedirs(self.hparams.output_dir, exist_ok=True)

    def __get_tokenizers(self):
        train_tokenizer = BertTokenizer.from_pretrained(
            self.hparams.tokenizer_name or self.hparams.pretrained_model)

        test_tokenizer = train_tokenizer  # By default, they are the same.

        if self.hparams.test_tokenizer_name is not None:
            test_tokenizer = BertTokenizer.from_pretrained(
                self.hparams.test_tokenizer_name)

        return (train_tokenizer, test_tokenizer)

    def __retrieve_eval_feature_set(self, results):
        examples = self.eval_examples
        features = self.eval_features

        if len(results) != len(features):
            # Working with a subset of the data (probably Fast Dev Run Mode)
            examples = [
                self.eval_examples[
                    self.eval_features_index[i.unique_id].example_index
                ] for i in results]

            features = [self.eval_features_index[i.unique_id] for i in results]

        return examples, features


class NLIModel(LightningModule):
    def __init__(self,
                 pretrained_model: str,
                 num_classes: int,
                 train_lexical_strategy: str,
                 test_lexical_strategy: str,
                 train_dataset: str,
                 test_dataset: str,
                 data_dir: str,
                 batch_size: int,
                 max_seq_length: int,
                 test_lexical_path: str = None,
                 tokenizer_name: str = None,
                 test_tokenizer_name: str = None,
                 **kwargs) -> None:

        super(NLIModel, self).__init__()

        self.save_hyperparameters()

        config = BertConfig.from_pretrained(
            pretrained_model,
            num_labels=num_classes)

        self.bert = BertForSequenceClassification.from_pretrained(
            pretrained_model, config=config)

        self.train_tokenizer, self.test_tokenizer = self.__get_tokenizers()
        self.metric = Accuracy(num_classes=num_classes)

        self.training_setup_performed = False
        self.test_setup_performed = False

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )

        return outputs

    def configure_optimizers(self):
        return Adam(self.bert.parameters(), lr=2e-5)

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = \
            batch['input_ids'], batch['attention_mask'], \
            batch['token_type_ids'], batch['label']

        outputs = self(input_ids, attention_mask, token_type_ids, labels)
        loss = outputs[0]
        logits = outputs[1]
        predicted = torch.argmax(logits, dim=-1)

        accuracy = self.metric(predicted, labels)

        logs = {'train_loss': loss, 'train_acc': accuracy}
        tensor_bar = {'train_acc': accuracy}

        return {'loss': loss, 'log': logs, 'progress_bar': tensor_bar}

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = \
            batch['input_ids'], batch['attention_mask'], \
            batch['token_type_ids'], batch['label']

        outputs = self(input_ids, attention_mask, token_type_ids)
        logits = outputs[0]
        predicted = torch.argmax(logits, dim=-1)

        accuracy = self.metric(predicted, labels)

        logs = {'test_acc': accuracy}
        return {'test_acc': accuracy, 'log': logs, 'progress_bar': logs}

    def test_epoch_end(self, outputs):
        accuracies = torch.stack([o['test_acc'] for o in outputs])
        mean_accuracy = accuracies.mean()

        return {'test_avg_accuracy': mean_accuracy}

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            self.hparams.batch_size,
            shuffle=True,
            num_workers=8
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            self.hparams.batch_size,
            shuffle=False,
            num_workers=8
        )

    def prepare_data(self) -> None:
        prepare_nli_dataset(
            self.hparams.train_dataset,
            'train',
            self.hparams.data_dir,
            self.train_tokenizer,
            self.hparams.max_seq_length)

        prepare_nli_dataset(
            self.hparams.test_dataset,
            'eval',
            self.hparams.data_dir,
            self.test_tokenizer,
            self.hparams.max_seq_length)

    def setup(self, stage: str):
        if stage == 'fit' and not self.training_setup_performed:
            setup_lexical_for_training(
                self.hparams.train_lexical_strategy,
                self.bert,
                self.train_tokenizer)

            self.train_dataset = load_nli_dataset(
                self.hparams.train_dataset,
                'train',
                self.hparams.data_dir,
                self.hparams.max_seq_length
            )

            self.training_setup_performed = True
        elif stage == 'test' and not self.test_setup_performed:
            setup_lexical_for_testing(
                self.hparams.test_lexical_strategy,
                self.bert,
                self.test_tokenizer,
                self.hparams.test_lexical_path
            )

            self.test_dataset = load_nli_dataset(
                self.hparams.test_dataset,
                'eval',
                self.hparams.data_dir,
                self.hparams.max_seq_length
            )

            self.test_setup_performed = True

    def __get_tokenizers(self):
        train_tokenizer = BertTokenizer.from_pretrained(
            self.hparams.tokenizer_name or self.hparams.pretrained_model)

        test_tokenizer = train_tokenizer  # By default, they are the same.

        if self.hparams.test_tokenizer_name is not None:
            test_tokenizer = BertTokenizer.from_pretrained(
                self.hparams.test_tokenizer_name)

        return (train_tokenizer, test_tokenizer)
