import os
import pytorch_lightning as pl
import torch

from crosslangt.lexical import setup_lexical_for_training
from torch.optim import Adam

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits, squad_evaluate)
from transformers.data.processors.squad import SquadResult


class QAFinetuneModel(pl.LightningModule):
    def __init__(self,
                 pretrained_model: str,
                 train_lexical_strategy: str,
                 learning_rate: float = 2e-5,
                 tokenizer_name: str = None,
                 max_answer_length: int = 30,
                 n_best_size: int = 20,
                 output_dir: str = './output',
                 **kwargs) -> None:

        super().__init__()

        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name
                                                       or pretrained_model)
        self.qa_model = AutoModelForQuestionAnswering.from_pretrained(
            pretrained_model)

        setup_lexical_for_training(self.hparams.train_lexical_strategy,
                                   self.qa_model, self.tokenizer)

    def forward(self, input_ids, attention_mask, token_type_ids,
                start_positions, end_positions, **kwargs):

        outputs = self.qa_model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                start_positions=start_positions,
                                end_positions=end_positions)

        return outputs

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]

        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss)

        return result

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        return self._eval_epoch_end(outputs, 'eval')

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self._eval_epoch_end(outputs, 'test')

    def _eval_step(self, batch, batch_idx):
        outputs = self(**batch)

        feature_unique_ids = batch['feature_id'].detach().cpu().tolist()
        start_scores, end_scores = (outputs[1:3] if 'start_positions' in batch
                                    else outputs[:2])

        results = []

        for i, unique_id in enumerate(feature_unique_ids):
            result = SquadResult(unique_id,
                                 start_scores[i].detach().cpu().tolist(),
                                 end_scores[i].detach().cpu().tolist())

            results.append(result)

        result = pl.EvalResult()
        result.squad_results = results

        return result

    def _eval_epoch_end(self, validation_step_output_result, stage):
        # Flatten the results accumulated per batch
        all_results = [
            r for b in validation_step_output_result.squad_results for r in b
        ]

        output_prediction_file = os.path.join(
            self.hparams.output_dir,
            f'predictions_epoch{self.current_epoch}-{stage}.json')

        output_nbest_file = os.path.join(
            self.hparams.output_dir,
            f'nbest_predictions_epoch{self.current_epoch}-{stage}.json')

        examples, features = self.datamodule.retrieve_examples_and_features(
            stage, all_results)

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
        eval_result = pl.EvalResult()

        for metric, value in results.items():
            eval_result.log(metric, torch.tensor(value))

        return eval_result

    def setup(self, stage: str):
        # We get a reference to the datamodule in use
        # so we can calculate SQuAD metrics properly
        self.datamodule = getattr(self.trainer, 'datamodule', None)


if __name__ == "__main__":
    from crosslangt.question_answering.data_qa import SquadDataModule
    from pytorch_lightning import Trainer

    os.makedirs('/tmp/data', exist_ok=True)

    squad_en_default = SquadDataModule(dataset_name='squad_en',
                                       tokenizer_name='bert-base-cased',
                                       data_dir='/tmp/data',
                                       batch_size=12,
                                       max_seq_length=384,
                                       max_query_length=64,
                                       doc_stride=128)

    model = QAFinetuneModel('bert-base-cased', 'none')
    trainer = Trainer(fast_dev_run=True)

    trainer.fit(model, squad_en_default)
