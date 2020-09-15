import logging
import torch

from pytorch_lightning import LightningModule
from pytorch_lightning.metrics import Accuracy
from torch import Tensor
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer)

from crosslangt.lexical import setup_lexical_for_training
from crosslangt.nli import load_nli_dataset, prepare_nli_dataset
from crosslangt.nli.dataprep_nli import NLI_DATASETS

logger = logging.getLogger(__name__)


class NLIFinetuneModel(LightningModule):
    def __init__(self,
                 pretrained_model: str,
                 num_classes: int,
                 train_lexical_strategy: str,
                 train_dataset: str,
                 data_dir: str,
                 batch_size: int,
                 max_seq_length: int,
                 eval_dataset: str = None,
                 tokenizer_name: str = None,
                 **kwargs) -> None:

        super(NLIFinetuneModel, self).__init__()

        self.save_hyperparameters()

        # Load model and tokenizer
        self.config = AutoConfig.from_pretrained(pretrained_model,
                                                 num_labels=num_classes)

        self.bert = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model, config=self.config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.tokenizer_name or self.hparams.pretrained_model)

        # Setup lexical for training
        setup_lexical_for_training(train_lexical_strategy, self.bert,
                                   self.tokenizer)

        # Using accuracy metric
        self.metric = Accuracy(num_classes=num_classes)

        # For compability, we keep eval_dataset as None by default
        # If not defined, we'll use the same as train_dataset
        # The split is defined later
        self.train_dataset_name = train_dataset
        self.eval_dataset_name = eval_dataset or train_dataset

        # Custom label mappings
        self.label_mappings = []

        # Define Keys for dataset generation
        self.__set_feature_keys()

    def add_label_mapping(self, model_predicted_label, target_label):
        self.label_mappings.append((model_predicted_label, target_label))

    def forward(self, **inputs):
        return self.bert(**inputs)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=2e-5)

    def training_step(self, batch, batch_idx):
        loss, accuracy = self._run_step(batch, batch_idx)

        logs = {'train_loss': loss, 'train_acc': accuracy}
        tensor_bar = {'train_acc': accuracy}

        return {'loss': loss, 'log': logs, 'progress_bar': tensor_bar}

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._run_step(batch, batch_idx, True, True)

        logs = {'val_acc': accuracy, 'val_loss': loss}
        return {
            'val_loss': loss,
            'val_acc': accuracy,
            'log': logs,
            'progress_bar': logs
        }

    def validation_epoch_end(self, outputs):
        return self._eval_epoch_end(outputs, 'val_')

    def test_step(self, batch, batch_idx):
        loss, accuracy = self._run_step(batch, batch_idx, True, True)

        logs = {'test_acc': accuracy, 'test_loss': loss}
        return {
            'test_loss': loss,
            'test_acc': accuracy,
            'log': logs,
            'progress_bar': logs
        }

    def test_epoch_end(self, outputs):
        return self._eval_epoch_end(outputs, 'test_')

    def _run_step(self,
                  batch,
                  batch_idx,
                  log: bool = False,
                  enable_label_remapping: bool = False):

        inputs = {}

        inputs['input_ids'] = batch['input_ids']
        inputs['attention_mask'] = batch['attention_mask']
        inputs['labels'] = batch['label']

        if self.config.model_type in ['bert', 'xlnet', 'albert']:
            inputs['token_type_ids'] = batch['token_type_ids']

        outputs = self(**inputs)

        loss, logits = outputs[:2]
        predicted = torch.argmax(logits, dim=-1)
        original = None

        if enable_label_remapping is True:
            predicted, original = self._apply_label_remapping(predicted)

        accuracy = self.metric(predicted, batch['label'])

        if log is True:
            self.__log_batch(inputs['input_ids'], inputs['labels'], predicted,
                             original)

        return loss, accuracy

    def _eval_epoch_end(self, outputs, prefix):
        acc_key = f'{prefix}acc'
        loss_key = f'{prefix}loss'

        accuracies = torch.stack([o[acc_key] for o in outputs])
        losses = torch.stack([o[loss_key] for o in outputs])

        mean_accuracy = accuracies.mean()
        mean_loss = losses.mean()

        results_dict = {
            f'{prefix}avg_accuracy': mean_accuracy,
            f'{prefix}avg_loss': mean_loss
        }

        return {**results_dict, 'log': results_dict}

    def _apply_label_remapping(self, original_predictions: torch.Tensor):
        masks = []
        predicted = original_predictions.clone()

        for original, target in self.label_mappings:
            masks.append((predicted == original, target))

        for mask, target in masks:
            predicted.masked_fill_(mask, target)

        return predicted, original_predictions

    def train_dataloader(self) -> DataLoader:
        dataset = load_nli_dataset(self.hparams.data_dir,
                                   self.train_dataset_name, 'train',
                                   self.hparams.max_seq_length, self.train_key)

        return DataLoader(dataset,
                          self.hparams.batch_size,
                          shuffle=True,
                          num_workers=8)

    def val_dataloader(self):
        dataset = load_nli_dataset(self.hparams.data_dir,
                                   self.eval_dataset_name, 'eval',
                                   self.hparams.max_seq_length, self.train_key)

        return DataLoader(dataset,
                          self.hparams.batch_size,
                          shuffle=False,
                          num_workers=8)

    def prepare_data(self) -> None:
        prepare_nli_dataset(dataset=self.train_dataset_name,
                            split='train',
                            data_dir=self.hparams.data_dir,
                            tokenizer=self.tokenizer,
                            max_seq_length=self.hparams.max_seq_length,
                            features_key=self.train_key)

        prepare_nli_dataset(dataset=self.eval_dataset_name,
                            split='eval',
                            data_dir=self.hparams.data_dir,
                            tokenizer=self.tokenizer,
                            max_seq_length=self.hparams.max_seq_length,
                            features_key=self.train_key)

    def __set_feature_keys(self):
        model_name = self.hparams.pretrained_model.split('/').pop()
        tokenizer_name = self.hparams.tokenizer_name or ''
        tokenizer_name = tokenizer_name.split('/').pop()

        self.train_key = f'{model_name}.{tokenizer_name}'

    def __log_batch(self, input_ids: Tensor, labels: Tensor, predicted: Tensor,
                    oringal_predictions: Tensor = None):

        if self.logger is not None and self.logger.experiment is not None \
           and self.logger.experiment.log_sample:

            if oringal_predictions is None:
                oringal_predictions = predicted

            for i, input_ids in enumerate(input_ids.tolist()):

                decoded = self.tokenizer.decode(input_ids)
                sentences = decoded.split(self.tokenizer.sep_token)

                self.logger.experiment.log_sample(
                    tag='nli-finetune',
                    sample_text=f'nli premise: {sentences[0].strip()}. '
                    f'hypothesis: {sentences[1].strip()}. '
                    f'expected: {labels[i]}. '
                    f'predicted: {predicted[i]}. '
                    f'original prediction: {oringal_predictions[i]}.',
                    global_step=self.global_step,
                    epoch=self.current_epoch
                )

        else:
            logger.warn(
                'No Logger has been specified. Samples will not be available.')
