from __future__ import absolute_import, division, print_function

import json
import datasets


# TODO(squad_v1_pt): BibTeX citation
_CITATION = """\
@inproceedings{Sayama2019,
    author = {Sayama, Helio Fonseca and Araujo, Anderson Vicoso and Fernandes, Eraldo Rezende},
    booktitle = {Proceedings - 2019 Brazilian Conference on Intelligent Systems, BRACIS 2019},
    doi = {10.1109/BRACIS.2019.00084},
    isbn = {9781728142531},
    keywords = {Dataset,Machine Reading Comprehension,Natural Language Processing},
    mendeley-groups = {NLP/Cross Lingual Transfer,NLP},
    month = {oct},
    pages = {443--448},
    publisher = {Institute of Electrical and Electronics Engineers Inc.},
    title = {{FaQuAD: Reading comprehension dataset in the domain of brazilian higher education}},
    year = {2019}
}
"""

_DESCRIPTION = """\
FaQuAD: A novel machine reading comprehension dataset in the domain of Brazilian higher education institutions
"""

_URL = "https://raw.githubusercontent.com/liafacom/faquad/master/data/"
_URLS = {
    "train": _URL + "train.json",
    "dev": _URL + "dev.json",
}


class Faquad(datasets.GeneratorBasedBuilder):

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
            homepage="https://github.com/liafacom/faquad",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO(squad_v1_pt): Downloads the data and defines the splits
        # dl_manager is a datasets.download.DownloadManager that can be used to
        # download and extract URLs
        urls_to_download = _URLS
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": downloaded_files["dev"]}),
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
