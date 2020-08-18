from crosslangt.lexical import setup_lexical_for_testing
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
from transformers.tokenization_utils import PreTrainedTokenizer
from crosslangt.nli.dataprep_nli import load_nli_dataset, prepare_nli_dataset
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


def run_nli_training(experiment_name: str,
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
                     seed: int = 123,
                     tokenizer_name: str = None):

    seed_everything(seed)

    base_exp_path = os.path.join(output_path, experiment_name)
    last_checkpoint = None

    # Try recover the last checkpoint for this experiment
    last_checkpoint = retrieve_last_checkpoint(base_exp_path)

    if last_checkpoint is not None:
        # We first try to recover the experiment from an unfinished run
        model = NLIModel.load_from_checkpoint(last_checkpoint)
    else:
        model = NLIModel(pretrained_model=pretrained_model,
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

    trainer = Trainer(resume_from_checkpoint=last_checkpoint,
                      gpus=gpus,
                      tpu_cores=tpu_cores,
                      checkpoint_callback=model_checkpoint,
                      accumulate_grad_batches=accumulate_grad,
                      max_epochs=max_epochs,
                      deterministic=True)

    trainer.fit(model)

    return trainer, model


def test_nli_checkpoint(test_experiment_key: str,
                        checkpoint_path: str,
                        test_dataset: str,
                        max_seq_length: int,
                        test_lexical_strategy: str,
                        test_lexical_path: str = None,
                        test_tokenizer_name: str = None,
                        prepare_data: bool = True,
                        gpus: int = 1,
                        tpu_cores: int = None,
                        seed: int = 123):

    seed_everything(seed)

    model = NLIModel.load_from_checkpoint(checkpoint_path)
    tokenizer = PreTrainedTokenizer.from_pretrained(
        test_tokenizer_name or model.hparams.pretrained_model)

    if prepare_data is True:

        prepare_nli_dataset(test_dataset,
                            'eval',
                            DEFAULT_DATA_DIR,
                            tokenizer,
                            max_seq_length,
                            test_experiment_key)

    dataset = load_nli_dataset(DEFAULT_DATA_DIR,
                               test_dataset,
                               'eval',
                               max_seq_length,
                               test_experiment_key)

    data_loader = DataLoader(dataset, shuffle=False, num_workers=8)

    # Apply the strategy for testing
    setup_lexical_for_testing(test_lexical_strategy, model.bert, tokenizer,
                              test_lexical_path)

    trainer = Trainer(gpus=gpus, tpu_cores=tpu_cores, deterministic=True)
    trainer.test(model, data_loader)

    return trainer, model
