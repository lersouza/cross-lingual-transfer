import os
import re
from typing import Tuple
import torch

from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torch.utils.data.dataloader import DataLoader

from crosslangt.lexical import SlicedEmbedding
from crosslangt.nli.datasets import NLIDataset
from crosslangt.nli.dataprep_nli import prepare_nli_dataset
from crosslangt.nli.modeling_nli import NLIFinetuneModel
from crosslangt.pretrain.models import LexicalTrainingModel


DEFAULT_DATA_DIR = './data'
DEFAULT_LOG_DIR = './logs'
DEFAULT_EXPERIMENT_LOCATION = './output'

NLI_CHECKPOINT_FORMAT = '{epoch}-{loss:.2f}'


def get_logger(experiment_name: str,
               logging_path: str,
               version: int = None):

    return TensorBoardLogger(logging_path, experiment_name, version)


def get_model_checkpoint_callback(experiment_path: str,
                                  checkpoint_file_format: str):

    callback = ModelCheckpoint(os.path.join(experiment_path,
                                            checkpoint_file_format),
                               save_top_k=-1,
                               period=1)

    return callback


def retrieve_last_checkpoint(experiment_path: str):
    path = Path(experiment_path)

    if not path.exists():
        return None  # No checkpoint exists

    all_checkpoints = [str(file) for file in path.glob('*.ckpt')]

    if len(all_checkpoints) == 0:
        return None

    # All checkpoints should separate vars by '-'
    # and the first one is the epoch (epoch=0, for instance)
    # to be sorted here.
    by_epoch = sorted(all_checkpoints,
                      key=lambda file: re.findall(r'epoch=(\d+)', file)[0],
                      reverse=True)

    return by_epoch[0]


def run_nli_finetune(experiment_name: str,
                     pretrained_model: str,
                     num_classes: int,
                     train_lexical_strategy: str,
                     train_dataset: str,
                     eval_dataset: str,
                     batch_size: int,
                     max_seq_length: int,
                     max_epochs: int,
                     accumulate_grad: int = 2,
                     gpus: int = 0,
                     tpu_cores: int = None,
                     output_path: str = DEFAULT_EXPERIMENT_LOCATION,
                     precision: int = 32,
                     seed: int = 123,
                     tokenizer_name: str = None,
                     experiment_version: int = None,
                     log_path: str = DEFAULT_LOG_DIR):

    seed_everything(seed)

    base_exp_path = os.path.join(output_path, experiment_name)
    last_checkpoint = None

    # Try recover the last checkpoint for this experiment
    last_checkpoint = retrieve_last_checkpoint(base_exp_path)

    if last_checkpoint is not None:
        # We first try to recover the experiment from an unfinished run
        model = NLIFinetuneModel.load_from_checkpoint(last_checkpoint)
    else:
        model = NLIFinetuneModel(pretrained_model=pretrained_model,
                                 num_classes=num_classes,
                                 train_lexical_strategy=train_lexical_strategy,
                                 train_dataset=train_dataset,
                                 eval_dataset=eval_dataset,
                                 data_dir=DEFAULT_DATA_DIR,
                                 batch_size=batch_size,
                                 max_seq_length=max_seq_length,
                                 tokenizer_name=tokenizer_name)

    model_checkpoint = get_model_checkpoint_callback(base_exp_path,
                                                     NLI_CHECKPOINT_FORMAT)

    logger = get_logger(experiment_name, log_path, experiment_version)

    trainer = Trainer(resume_from_checkpoint=last_checkpoint,
                      logger=logger,
                      gpus=gpus,
                      tpu_cores=tpu_cores,
                      checkpoint_callback=model_checkpoint,
                      accumulate_grad_batches=accumulate_grad,
                      max_epochs=max_epochs,
                      precision=precision,
                      deterministic=True)

    trainer.fit(model)

    return trainer, model


def prepare_model_for_testing(checkpoint_path: str,
                              lexical_checkpoint: str = None,
                              always_use_finetuned_lexical: bool = False):

    model = NLIFinetuneModel.load_from_checkpoint(checkpoint_path)
    model_training = model.hparams.train_lexical_strategy

    if model_training == 'none' or always_use_finetuned_lexical is True:
        return model

    lexical_model = LexicalTrainingModel.load_from_checkpoint(
        lexical_checkpoint)

    # Setup Target Lexical Tokenizer
    model.tokenizer = lexical_model.tokenizer

    if model_training == 'freeze-all':
        model.bert.set_input_embeddings(
            lexical_model.bert.get_input_embeddings())

    elif model_training == 'freeze-nonspecial':
        # In thise case, embedding should be a sliced embedding
        # So, we just take the parts we want to form a new Embedding

        # Special tokens from the finetuned model
        model_weights = model.bert.get_input_embeddings().get_first_weigths()

        # Other tokens (language specific) from the lexical aligned model
        lexical_embeddings = lexical_model.bert.get_input_embeddings()
        target_weights = lexical_embeddings.get_second_weigths()

        tobe = SlicedEmbedding(model_weights,
                               target_weights, True,
                               True)  # For testing, both are freezed

        model.bert.set_input_embeddings(tobe)

    return model


def test_nli_checkpoint(testing_key: str,
                        checkpoint_path: str,
                        data_dir: str,
                        dataset_name: str,
                        dataset_split: str,
                        gpus: int = 1,
                        always_use_finetuned_lexical: bool = False,
                        lexical_checkpoint: str = None,
                        label_remappings: Tuple[int, int] = None,
                        test_output_path: str = DEFAULT_EXPERIMENT_LOCATION,
                        seed: int = 123,
                        log_path: str = DEFAULT_LOG_DIR,
                        cleanup: bool = True):

    seed_everything(seed)

    model = prepare_model_for_testing(
        checkpoint_path=checkpoint_path,
        lexical_checkpoint=lexical_checkpoint,
        always_use_finetuned_lexical=always_use_finetuned_lexical)

    if label_remappings is not None:
        for source, target in label_remappings:
            model.add_label_mapping(source, target)

    dataset_path = prepare_nli_dataset(
        dataset=dataset_name,
        split=dataset_split,
        data_dir=data_dir,
        tokenizer=model.tokenizer,
        max_seq_length=model.hparams.max_seq_length,
        features_key=testing_key)

    dataset = NLIDataset(torch.load(dataset_path))
    data_loader = DataLoader(dataset,
                             batch_size=model.hparams.batch_size,
                             shuffle=False,
                             num_workers=8)

    trainer = Trainer(gpus=gpus,
                      logger=get_logger(testing_key, log_path))

    metrics = trainer.test(model, test_dataloaders=data_loader)

    # Do some cleanup, since the model and dataset uses a lot of memory
    if cleanup is True:
        del model
        del trainer
        del dataset

    return metrics
