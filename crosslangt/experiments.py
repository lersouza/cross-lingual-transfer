import re
import os
from pathlib import Path

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import exp

from crosslangt.models import NLIModel

DEFAULT_DATA_DIR = './data'
DEFAULT_EXPERIMENT_LOCATION = './output'

NLI_CHECKPOINT_FORMAT = '{epoch}-{loss:.2f}'


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


def run_nli_experiment(experiment_name: str,
                       pretrained_model: str,
                       num_classes: int,
                       train_lexical_strategy: str,
                       test_lexical_strategy: str,
                       test_lexical_path: str,
                       train_dataset: str,
                       test_dataset: str,
                       batch_size: int,
                       max_seq_length: int,
                       max_epochs: int,
                       accumulate_grad: int = 2,
                       gpus: int = 0,
                       model_checkpoint: str = None,
                       output_path: str = DEFAULT_EXPERIMENT_LOCATION,
                       run_training: bool = True,
                       run_test: bool = True,
                       seed: int = 123,
                       train_tokenizer_name: str = None,
                       test_tokenizer_name: str = None):
    seed_everything(seed)

    base_exp_path = os.path.join(output_path, experiment_name)
    last_checkpoint = None

    # Try recover the last checkpoint for this experiment
    last_checkpoint = retrieve_last_checkpoint(base_exp_path)

    if last_checkpoint is not None:
        # We first try to recover the experiment from an unfinished run
        model = NLIModel.load_from_checkpoint(last_checkpoint)
    elif model_checkpoint is not None:
        # If no unfinished run is present, but user asks for a particular
        # model, we load it and respect the experiment parameters.
        # Is it a transfer learning task, perhaps?
        model = NLIModel.load_from_checkpoint(
            model_checkpoint,
            pretrained_model=pretrained_model,
            num_classes=num_classes,
            train_lexical_strategy=train_lexical_strategy,
            test_lexical_strategy=test_lexical_strategy,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            data_dir=DEFAULT_DATA_DIR,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            tokenizer_name=train_tokenizer_name,
            test_lexical_path=test_lexical_path,
            test_tokenizer_name=test_tokenizer_name)
    else:
        model = NLIModel(pretrained_model=pretrained_model,
                         num_classes=num_classes,
                         train_lexical_strategy=train_lexical_strategy,
                         test_lexical_strategy=test_lexical_strategy,
                         train_dataset=train_dataset,
                         test_dataset=test_dataset,
                         data_dir=DEFAULT_DATA_DIR,
                         batch_size=batch_size,
                         max_seq_length=max_seq_length,
                         tokenizer_name=train_tokenizer_name,
                         test_lexical_path=test_lexical_path,
                         test_tokenizer_name=test_tokenizer_name)

    model_checkpoint = get_model_checkpoint_callback(base_exp_path,
                                                     NLI_CHECKPOINT_FORMAT)

    trainer = Trainer(resume_from_checkpoint=last_checkpoint,
                      gpus=gpus,
                      checkpoint_callback=model_checkpoint,
                      accumulate_grad_batches=accumulate_grad,
                      max_epochs=max_epochs,
                      deterministic=True)

    if run_training is True:
        trainer.fit(model)

    if run_test is True:
        trainer.test(model)


def test_nli_checkpoint(checkpoint_path: str,
                        test_dataset: str,
                        test_lexical_strategy: str,
                        test_lexical_path: str = None,
                        test_tokenizer_name: str = None,
                        prepare_data: bool = True,
                        gpus: int = 1):

    model = NLIModel.load_from_checkpoint(
            checkpoint_path,
            test_dataset=test_dataset,
            test_lexical_strategy=test_lexical_strategy,
            data_dir=DEFAULT_DATA_DIR,
            test_lexical_path=test_lexical_path,
            test_tokenizer_name=test_tokenizer_name)

    if prepare_data is True:
        model.prepare_data()

    trainer = Trainer(gpus=gpus, deterministic=True)
    trainer.test(model)


if __name__ == '__main__':
    run_nli_experiment('test_1', 'bert-base-cased', 3, 'none', 'original',
                       None, 'assin2', 'assin2', 2, 60, 1)
