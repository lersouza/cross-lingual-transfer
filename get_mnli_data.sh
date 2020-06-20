#!/bin/bash

DATASETS_FOLDER=${1:-"data/"}

MNLI_DATASET_FOLDER="$DATASETS_FOLDER/mnli/"
MNLI_URL="https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce"

mkdir -p $MNLI_DATASET_FOLDER

echo "[1] Downloading MNLI Dataset"
if [ ! -f "$MNLI_DATASET_FOLDER/dataset.zip" ]; then
  curl --output $MNLI_DATASET_FOLDER/dataset.zip $MNLI_URL
else
  echo "   zip file already downloaded"
fi
echo ""

echo "[2] Extracting ..."
unzip $MNLI_DATASET_FOLDER/dataset.zip -d $MNLI_DATASET_FOLDER/
echo ""

echo "[3] Clean up ..."
mv $MNLI_DATASET_FOLDER/MNLI/* $MNLI_DATASET_FOLDER
rm -rf $MNLI_DATASET_FOLDER/MNLI/
echo ""
