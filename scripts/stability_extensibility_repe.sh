#!/bin/bash

# custom config
DATA=data
TRAINER=$1
DATASET=$2
CFG=$3  # rn50, rn101, vit_b32 or vit_b16
LAMBDA=$4
RETRIEVED_NUM=$5
TRIALS=$6


python main.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${CFG}.yaml \
--target-label-dir output/${TRAINER}/${CFG}/${DATASET}/openness/repe_prompt_lambda${LAMBDA}_K${RETRIEVED_NUM} \
--eval-only \
--incremental-evaluation \
--trials ${TRIALS} \
--repe \
--shift-lambda ${LAMBDA} \
--retrieved-num ${RETRIEVED_NUM}

# CUDA_VISIBLE_DEVICES=0 bash scripts/stability_extensibility_repe.sh ZeroshotCLIP2 my_cifar100 vit_b32 0.25 100 3
# CUDA_VISIBLE_DEVICES=0 bash scripts/stability_extensibility_repe.sh ZeroshotCLIP2 imagenet_entity13 vit_b32 0.25 100 3
# CUDA_VISIBLE_DEVICES=0 bash scripts/stability_extensibility_repe.sh ZeroshotCLIP2 imagenet_living17 vit_b32 0.25 100 3
