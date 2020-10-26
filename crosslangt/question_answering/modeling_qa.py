import os
import pytorch_lightning as pl
import torch
from transformers.modeling_bert import BertForQuestionAnswering

from crosslangt.lexical import setup_lexical_for_training
from torch.optim import Adam

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers.data.processors.squad import SquadResult

from deeppavlov.metrics.squad_metrics import squad_v1_f1, squad_v1_exact_match


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

        start_scores, end_scores = (outputs[1:3] if 'start_positions' in batch
                                    else outputs[:2])

        max_context_length = torch.sum(batch['token_type_ids'], dim=-1)

        pred_starts = torch.argmax(start_scores, dim=-1)
        pred_ends = torch.argmax(end_scores, dim=-1)

        zero = torch.tensor(0).long()

        pred_starts.masked_fill_(pred_starts > max_context_length, zero)
        pred_ends.masked_fill_(pred_ends > max_context_length, zero)

        result = pl.EvalResult()

        result.predicted_starts = pred_starts.detach().cpu().tolist()
        result.predicted_ends = pred_ends.detach().cpu().tolist()

        return result

    def _eval_epoch_end(self, validation_step_output_result, stage):
        # Flatten the results accumulated per batch
        predicted_starts = [
            r for b in validation_step_output_result.predicted_starts
            for r in b
        ]
        predicted_ends = [
            r for b in validation_step_output_result.predicted_ends for r in b
        ]

        answers, predicted = self.datamodule.post_process_data(
            predicted_starts, predicted_ends, stage)

        self._log_predictions(answers, predicted)

        f1_score = squad_v1_f1(answers, predicted)
        em_score = squad_v1_exact_match(answers, predicted)

        eval_result = pl.EvalResult()

        eval_result.log(f'{stage}-f1', torch.tensor(f1_score))
        eval_result.log(f'{stage}-em', torch.tensor(em_score))

        return eval_result

    def setup(self, stage: str):
        # We get a reference to the datamodule in use
        # so we can calculate SQuAD metrics properly
        self.datamodule = getattr(self.trainer, 'datamodule', None)

    def _log_predictions(self, groud_truth, predictions):
        if self.logger and self.logger.experiment \
                and self.logger.experiment.log_sample:

            for i, (truth, pred) in enumerate(zip(groud_truth, predictions)):
                truth = ';'.join(groud_truth)
                sample = f'Sample: {i}. Truth: {truth}. Prediction: {pred}.'
                self.logger.experiment.log_sample(tag='qa',
                                                  sample_text=sample,
                                                  global_step=self.global_step,
                                                  epoch=self.current_epoch)
