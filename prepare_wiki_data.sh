#!/bin/bash

ROOT=$(pwd)
SCRIPTS="$ROOT/scripts"

LANG=$1
TARGET=${2:-"$ROOT/output"}
SEED=${3:-54321}

WIKI_DIR="$TARGET/datasets/$LANG/bz2"
WIKI_FILE="${LANG}wiki-latest-pages-articles.xml.bz2"
DATASET_DIR="$TARGET/datasets/$LANG/txt"
TOKENIZER_DIR="$TARGET/tokenizers/$LANG"

WIKI_EXTRACTOR=$ROOT/external/wikiextractor/WikiExtractor.py

mkdir -p $DATASET_DIR
mkdir -p $TOKENIZER_DIR

echo "Dataset Language: $LANG"
echo "Target directory: $TARGET"
echo "Scripts Directory: $SCRIPTS"
echo "Random Seed: $SEED"
echo ""

echo "[1] Downloading Wikipedia Dataset"
python $SCRIPTS/download_wiki_dump.py $LANG $WIKI_DIR

echo ""

echo "[2] Extracting Wikipedia Data"
if [ ! -f $DATASET_DIR/wiki.all ]; then
  python $WIKI_EXTRACTOR $WIKI_DIR/$WIKI_FILE --processes 8 -q -o - \
    | sed "/^\s*\$/d" \
    | grep -v "^<doc id=" \
    | sed -e "s/<\/doc>\$//g" \
    | python $SCRIPTS/split_paragraph.py $LANG \
    > $DATASET_DIR/wiki.all
else
  echo "$DATASET_DIR/wiki.all already exists. Skipping..."
fi
echo ""

echo "[3] Split dataset into train, val and test"
DOC_COUNT=$(awk -v RS= 'END{print NR}' output/datasets/pt/txt/wiki.all)
TRAIN_SPLIT=$((DOC_COUNT*90/100))
VALID_SPLIT=$((DOC_COUNT*5/100))
TEST_SPLIT=$((DOC_COUNT-(TRAIN_SPLIT+VALID_SPLIT)))

echo "- Total Documents: $DOC_COUNT"
echo "- Splits: $TRAIN_SPLIT for train, $VALID_SPLIT for valid, $TEST_SPLIT for test."
echo ""

echo "Shuffle dataset ..."
# Shuffle based on https://unix.stackexchange.com/questions/406382/multi-line-file-shuffle
# for shuffling full documents, not just lines
awk -v seed=$SEED '
  BEGIN{srand(seed); n=rand()}
  {print n, NR, $0}
  !NF {n=rand()}
  END {if (NF) print n, NR+1, ""}' $DATASET_DIR/wiki.all \
    | sort -nk1 -k2 \
    | cut -d' ' -f3- \
    > $DATASET_DIR/wiki.all.shuffled
echo "Shuffled dataset: $DATASET_DIR/wiki.all.shuffled"
echo ""

echo "Splitting ..."

TRAIN_START=1
TRAIN_END=$TRAIN_SPLIT

VALID_START=$TRAIN_SPLIT
VALID_END=$((TRAIN_SPLIT+VALID_SPLIT))

TEST_START=$VALID_END
TEST_END=$((VALID_END+TEST_SPLIT))

awk -v min=$TRAIN_START \
    -v max=$TRAIN_END \
    -v RS= \
    '{ if(NR>min&&NR<=max) { print $0, "\n" } }' $DATASET_DIR/wiki.all.shuffled > $DATASET_DIR/wiki.train

awk -v min=$VALID_START \
    -v max=$VALID_END \
    -v RS= \
    '{ if(NR>min&&NR<=max) { print $0, "\n" } }' $DATASET_DIR/wiki.all.shuffled > $DATASET_DIR/wiki.valid

awk -v min=$TEST_START \
    -v max=$TEST_END \
    -v RS= \
    '{ if(NR>min&&NR<=max) { print $0, "\n" } }' $DATASET_DIR/wiki.all.shuffled > $DATASET_DIR/wiki.test

echo "Removing tmp shuffled file..."
rm $DATASET_DIR/wiki.all.shuffled
echo ""


echo "[4] Splitting large files"
SPLIT_SIZE=300000

if (($TRAIN_SPLIT > $SPLIT_SIZE )); then
  echo "- Splitting training data"
  awk -v dir=$DATASET_DIR \
      -v size=$SPLIT_SIZE \
      -v RS= \
      'NR%size==1{x="wiki.train."++i;}{print $0, "\n" > (dir "/" x)}' $DATASET_DIR/wiki.train

  rm $DATASET_DIR/wiki.train
fi

if (($VALID_SPLIT > $SPLIT_SIZE )); then
  echo "- Splitting validation data"
  awk -v dir=$DATASET_DIR \
      -v size=$SPLIT_SIZE \
      -v RS= \
      'NR%size==1{x="wiki.valid."++i;}{print $0, "\n" > (dir "/" x)}' $DATASET_DIR/wiki.valid

  rm $DATASET_DIR/wiki.valid
fi

if (($TEST_SPLIT > $SPLIT_SIZE )); then
  echo "- Splitting test data"
  awk -v dir=$DATASET_DIR \
      -v size=$SPLIT_SIZE \
      -v RS= \
      'NR%size==1{x="wiki.test."++i;}{print $0, "\n" > (dir "/" x)}' $DATASET_DIR/wiki.test

  rm $DATASET_DIR/wiki.test
fi
