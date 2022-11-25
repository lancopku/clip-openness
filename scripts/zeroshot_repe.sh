#!/bin/bash

# custom config
DATA=data/
TRAINER=$1
DATASET=$2
CFG=$3  # rn50, rn101, vit_b32 or vit_b16
LAMBDA=$4
RETRIEVED_NUM=$5

python main.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${CFG}.yaml \
--output-dir output/${TRAINER}_repe/${CFG}/${DATASET} \
--repe \
--shift-lambda ${LAMBDA} \
--retrieved-num ${RETRIEVED_NUM} \
--eval-only

# CUDA_VISIBLE_DEVICES=0 sh scripts/zeroshot_repe.sh ZeroshotCLIP2 my_cifar100 vit_b32 0.25 100
# CUDA_VISIBLE_DEVICES=0 sh scripts/zeroshot_repe.sh ZeroshotDeCLIP2 my_cifar100 vit_b32_declip 0.25 100
# CUDA_VISIBLE_DEVICES=0 sh scripts/zeroshot_repe.sh ZeroshotSLIP2 my_cifar100 vit_b16_slip_ep100 0.25 100