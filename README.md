# Transfer Learning from English to Portuguese

This repository contains the source code for some experiments with cross lingual transfer learning, from English to Portuguese, using BERT.
It is organized as follows:

- *notebooks*: Notebooks used to perform some training/demo of the concepts explored during cross-lingual transfer. It contains an explanation on how I've trained BERT Word Embeddings (originally pre-trained on English corpus) on Wiki PT.

- *crosslangt*: Contains functions used to perform the cross lingual transfer in tasks such as Question-Answering and Natural Language Inference. It also contains a functions to generate pre-training data for BERT based on original BERT source code (https://github.com/google-research/bert). The latter is an ongoing work to better train the PT Word embeddings and study the impact of the pre-trained procedure evaluation during Language transfering.

- *scripts*: Contains python scripts to finetune BERT to SQuAD and MNLI Datasets, as well as pre-process Wikipedia data for BERT pre-training (on-going work). The finetune scripts are heavily based on [Huggingface transformers' examples](https://github.com/huggingface/transformers), with some adaptations to manipulate word embeddings.

## The Experiment

The experiment performed here is based on Artetxe, Ruder and Yogatama (2019) work, that shows that cross-lingual transfer learning can be performed with low cost, at lexical level.

The idea follows to basic steps:

1. We finetune the Word Embeddings of a pre-trained BERT to an unsupervised corpus of the target language of choice (Portuguese, in my case). The rest of the Transformer body remains freezed during the fine tuning.

2. The English pre-trained BERT is then finetuned to a downstream task in an English corpus. During this procedure, the Word Embeddings are freezed (so they remain compatible with the ones trained in the previous step). This finetune BERT can, then, be used on the target language Zero-Shot tasks by only switching its vocabulary and Word Embeddings.

For step 2, we run the following Scripts in SQuAD v1.1 Dataset and MNLI Datasets:

### SQuAD

Acquires SQuAD v1.1 Data:

```sh
./get_squad_data.sh
```

Fine-tune BERT en model to SQuAD English:

```sh
python scripts/finetune_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-cased \
  --freeze_lexical \
  --do_train \
  --do_eval \
  --train_file data/squad/train-v1.1.json \
  --predict_file data/squad/dev-v1.1.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 2e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --seed 123 \
  --save_steps 5000 \
  --output_dir output/debug_squad/ \
  --overwrite_output_dir
  ```

### MNLI

Acquires SQuAD v1.1 Data:

```sh
./get_mnli_data.sh
```

Finetune BERT on MNLI Dataset:

```sh
python scripts/finetune_mnli.py \
  --model_name_or_path bert-base-cased \
  --freeze_lexical \
  --task_name mnli \
  --do_train \
  --do_eval \
  --do_predict \
  --data_dir data/mnli/ \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 2.0 \
  --save_total_limit 5 \
  --seed 123 \
  --output_dir output/mnli/ \
  --overwrite_output_dir
  ```

## References

Artetxe, M., Ruder, S., & Yogatama, D. (2019). On the Cross-lingual Transferability of Monolingual Representations. http://arxiv.org/abs/1910.11856