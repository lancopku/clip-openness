#!/bin/bash

# custom config
DATA=data
TRAINER=$1  # ZeroshotCLIP, ZeroshotCLIP2, ZeroshotDeCLIP, ZeroshotSLIP, ZeroshotFILIP
DATASET=$2
CFG=$3  # rn50, rn101, vit_b32, vit_b16, vit_b32_declip, vit_b16_slip_ep100, vit_b32_filip
TRIALS=$4  # num of trials

python main.py \
--root ${DATA} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${CFG}.yaml \
--target-label-dir output/${TRAINER}/${CFG}/${DATASET}/openness/base_prompt/ \
--eval-only \
--incremental-evaluation \
--trials ${TRIALS}

# intra-dataset expansion
# CUDA_VISIBLE_DEVICES=0 bash scripts/stability_extensibility.sh ZeroshotCLIP my_cifar100 vit_b32 3
# CUDA_VISIBLE_DEVICES=0 bash scripts/stability_extensibility.sh ZeroshotCLIP imagenet_entity13 vit_b32 3
# CUDA_VISIBLE_DEVICES=0 bash scripts/stability_extensibility.sh ZeroshotCLIP imagenet_living17 vit_b32 3

# dataset-level expansion
# CUDA_VISIBLE_DEVICES=0 bash scripts/stability_extensibility.sh ZeroshotCLIP flowers_pets_cars vit_b32 2
# CUDA_VISIBLE_DEVICES=0 bash scripts/stability_extensibility.sh ZeroshotCLIP cifar100_caltech101_sun397 vit_b32 2
# CUDA_VISIBLE_DEVICES=0 bash scripts/stability_extensibility.sh ZeroshotCLIP cifar10_cifar100_imagenet vit_b32 2