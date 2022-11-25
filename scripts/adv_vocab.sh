#!/bin/bash

# custom config
DATA=data/
TRAINER=$1
DATASET=$2
CFG=$3  # config file
ADV_LABEL_DIR=$4  # dir of instance-level adv label
ADV_VOCAB_FILE=$5

ADV_LABEL_DIR=output/${TRAINER}/${CFG}/${DATASET}/${ADV_LABEL_DIR}/test
ADV_VOCAB_FILE=data/vocab/${ADV_VOCAB_FILE}

python main.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${CFG}.yaml \
--output-dir ${ADV_LABEL_DIR} \
--eval-only \
--reassign-test-adv-cn \
--adv-vocab-file ${ADV_VOCAB_FILE}

# CUDA_VISIBLE_DEVICES=0 sh scripts/adv_vocab.sh ZeroshotCLIP my_cifar100 vit_b16 adv_class_wordnet_noun_vocab wordnet-noun/wordnet-vocab-noun-aa