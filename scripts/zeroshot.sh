#!/bin/bash

# custom config
DATA=~/CoOp/data/
TRAINER=$1
DATASET=$2
CFG=$3  # rn50, rn101, vit_b32 or vit_b16

python main.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${CFG}.yaml \
--output-dir output/${TRAINER}/${CFG}/${DATASET} \
--eval-only

# CUDA_VISIBLE_DEVICES=0 sh scripts/zeroshot.sh ZeroshotCLIP2 my_cifar100 vit_b32