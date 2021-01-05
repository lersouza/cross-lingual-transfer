from __future__ import absolute_import, division, print_function

import json
import os

import datasets


_DESCRIPTION = """\
SberSQuAD
"""

_URL = "http://files.deeppavlov.ai/datasets/sber_squad_clean-v1.1.tar.gz"


class Sbersquad(datasets.GeneratorBasedBuilder):

    # Folowwing Squad version, for schema compat
    VERSION = datasets.Version("1.1.0")

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "title": datasets.Value("string"),
                    "context": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.features.Sequence(
                        {
                            "text": datasets.Value("string"),
                            "answer_start": datasets.Value("int32"),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="https://sdsj.sberbank.ai/static/2017/ru/contest.html",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(squad_v1_pt): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        downloaded_files = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": os.path.join(downloaded_files, 'train-v1.1.json')}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": os.path.join(downloaded_files, 'dev-v1.1.json')}),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
            for example in data["data"]:
                title = example.get("title", "").strip()
                for paragraph in example["paragraphs"]:
                    context = paragraph["context"].strip()
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]

                        answer_starts = [answer["answer_start"] for answer
                                         in qa["answers"]]

                        answers = [answer["text"].strip() for answer
                                   in qa["answers"]]

                        yield id_, {
                            "title": title,
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {
                                "answer_start": answer_starts,
                                "text": answers,
                            },
                        }
