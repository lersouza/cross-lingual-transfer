import csv
import io
import os

from typing import Optional, Union
from pytorch_lightning.loggers.csv_logs import CSVLogger, ExperimentWriter


class ExtendedExperimentWriter(ExperimentWriter):
    """
    Adds more functionality to the `ExperimentWriter` class
    in order to log texts for Cross Lingual Experiments.
    """

    SAMPLES_LOG_FILE = 'samples.csv'

    def __init__(self, log_dir: str) -> None:
        super().__init__(log_dir)

        self.samples_file_path = os.path.join(log_dir, self.SAMPLES_LOG_FILE)
        self.samples = []

    def log_sample(self,
                   tag: str,
                   sample_text: str,
                   global_step: int = 0,
                   epoch: int = 0):

        self.samples.append({
            'tag': tag,
            'global_step': global_step,
            'epoch': epoch,
            'sample': sample_text
        })

    def save(self) -> None:
        super().save()

        if self.samples:
            samples_fields = list(self.samples[0].keys())

            with io.open(self.samples_file_path, 'w', newline='') as f:
                self.writer = csv.DictWriter(f, fieldnames=samples_fields)
                self.writer.writeheader()
                self.writer.writerows(self.samples)


class CrossLangLogger(CSVLogger):
    def __init__(self, save_dir: str, name: Optional[str] = "default",
                 version: Optional[Union[int, str]] = None) -> None:

        super().__init__(save_dir, name, version)

        self._experiment = ExtendedExperimentWriter(self.log_dir)
