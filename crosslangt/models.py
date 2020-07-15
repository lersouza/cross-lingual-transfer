import os
import torch

from pytorch_lightning import LightningModule
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from transformers import BertForQuestionAnswering, BertTokenizer
from transformers.data.metrics.squad_metrics import compute_predictions_logits, squad_evaluate
from transformers.data.processors.squad import SquadFeatures, SquadResult

from crosslangt.dataprep import (load_question_answer_dataset,
                                 prepare_question_answer_dataset)


class QuestionAnsweringModel(LightningModule):
    """
    A model for Question-Answering. This model is suitable for training
    and finetuning with SQuAD 1.1 compatible datasets.
    """

    def __init__(self,
                 pretrained_model: str,
                 dataset: str,
                 data_dir: str,
                 batch_size: int,
                 max_seq_length: int,
                 max_query_length: int,
                 doc_stride: int,
                 output_dir: str,
                 n_best_size: int = 20,
                 max_answer_length: int = 30,
                 **kwargs) -> None:

        super(QuestionAnsweringModel, self).__init__()

        self.save_hyperparameters()

        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model)
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

    def validation_step(self, batch, batch_idx):
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

    def validation_epoch_end(self, outputs):
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
            self.tokenizer,
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

    def val_dataloader(self) -> DataLoader:
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
            self.tokenizer,
            self.hparams.max_seq_length,
            self.hparams.doc_stride,
            self.hparams.max_query_length
        )

        prepare_question_answer_dataset(
            self.hparams.dataset,
            'eval',
            self.hparams.data_dir,
            self.tokenizer,
            self.hparams.max_seq_length,
            self.hparams.doc_stride,
            self.hparams.max_query_length
        )

    def setup(self, stage: str):
        train_data = load_question_answer_dataset(
            self.hparams.dataset,
            'train',
            self.hparams.data_dir,
            self.hparams.max_seq_length
        )
        self.train_dataset = train_data['dataset']

        eval_data = load_question_answer_dataset(
            self.hparams.dataset,
            'eval',
            self.hparams.data_dir,
            self.hparams.max_seq_length
        )

        self.eval_dataset = eval_data['dataset']
        self.eval_examples = eval_data['examples']
        self.eval_features = eval_data['features']
        self.eval_features_index = {f.unique_id: f for f in self.eval_features}

        os.makedirs(self.hparams.output_dir, exist_ok=True)

    def __retrieve_eval_feature_set(self, results):
        examples = self.eval_examples
        features = self.eval_features

        if len(results != features):
            # Working with a subset of the data (probably Fast Dev Run Mode)
            examples = [
                self.eval_examples[
                    self.eval_features_index[i.unique_id].example_index
                ] for i in results]

            features = [self.eval_features_index[i.unique_id] for i in results]

        return examples, features


if __name__ == '__main__':
    from pytorch_lightning import Trainer

    trainer = Trainer(fast_dev_run=True)
    model = QuestionAnsweringModel(
        'bert-base-cased',
        'faquad',
        '/tmp/datasets/faquad',
        12, 256, 64, 128, '/tmp/output/faquad')

    trainer.fit(model)
