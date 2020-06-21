#!/bin/bash

DATASETS_FOLDER=${1:-"data/"}
!mkdir -p squad_dataset/

SQUAD_DATASET_FOLDER="$DATASETS_FOLDER/squad/"

SQUAD_TRAIN_URL="https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
SQUAD_EVAL_URL="https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"

mkdir -p $SQUAD_DATASET_FOLDER

echo "[1] Downloading SQuAD 1.1 Dataset"
curl --output $SQUAD_DATASET_FOLDER/train-v1.1.json $SQUAD_TRAIN_URL
curl --output $SQUAD_DATASET_FOLDER/dev-v1.1.json $SQUAD_EVAL_URL
echo ""

