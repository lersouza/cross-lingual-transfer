from crosslangt.lexical import setup_lexical_for_training
import os
import pytorch_lightning as pl
import torch

from torch.optim import Adam

from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits, squad_evaluate)
from transformers.data.processors.squad import SquadResult


class QAFinetuneModel(pl.LightningModule):
    def __init__(self,
                 pretrained_model: str,
                 train_lexical_strategy: str,
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

        setup_lexical_for_training(
            self.hparams.train_lexical_strategy,
            self.qa_model,
            self.train_tokenizer)

    def forward(self, input_ids, attention_mask, token_type_ids,
                start_positions, end_positions, **kwargs):

        outputs = self.qa_model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                start_positions=start_positions,
                                end_positions=end_positions)

        return outputs

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=2e-5)

    def training_step(self, batch, batch_idx):
        loss, _ = self(**batch)
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

        feature_unique_ids = batch['feature_id']
        start_scores, end_scores = outputs[1:3]

        results = []

        for i, unique_id in enumerate(feature_unique_ids):
            result = SquadResult(unique_id,
                                 start_scores[i].detach().cpu().tolist(),
                                 end_scores[i].detach().cpu().tolist())

            results.append(result)

        result = pl.EvalResult()
        result.predictions = results

        return result

    def _eval_epoch_end(self, validation_step_output_result, stage):
        all_results = validation_step_output_result.predictions

        output_prediction_file = os.path.join(
            self.hparams.output_dir,
            f'predictions_epoch{self.current_epoch}.json')

        output_nbest_file = os.path.join(
            self.hparams.output_dir,
            f'nbest_predictions_epoch{self.current_epoch}.json')

        examples, features = self.datamodule.retrieve_examples_and_features(
            stage,
            all_results
        )

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

        validation_step_output_result.exact = torch.tensor(results['exact'])
        validation_step_output_result.f1 = torch.tensor(results['f1'])
        validation_step_output_result.total = torch.tensor(results['total'])

        return validation_step_output_result
