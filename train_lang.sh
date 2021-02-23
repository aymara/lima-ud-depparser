#!/bin/bash

set -u

FT_FILE=$1
UD_PATH=$2
UD_LANG=$3
SAVE_TO=$4
FT_SRC=$5

UD_CORP=$UD_LANG

mkdir -p $SAVE_TO

cat $UD_PATH/UD_$UD_LANG/*.conllu | gawk -F $'\t' '{ print $2 }' | perl -CSDA -plE 's/\s+//g' | perl to_lower.pl | LC_COLLATE=POSIX sort | LC_COLLATE=POSIX uniq > $SAVE_TO/all_words.txt

echo Found `wc -l $SAVE_TO/all_words.txt` different words ...

$FT_SRC/fasttext print-word-vectors $FT_FILE < $SAVE_TO/all_words.txt > $SAVE_TO/all_words.vectors

echo Found `wc -l $SAVE_TO/all_words.vectors` embeddings ...

ts=`date '+%d_%H_%M'`;
python3 main.py \
                -u $UD_PATH \
                -t $UD_CORP \
                -e $SAVE_TO/all_words.vectors \
                -l $UD_LANG \
                -p $SAVE_TO/

