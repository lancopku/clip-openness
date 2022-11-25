#!/bin/bash

# custom config
DATA=data
TRAINER=CoOp

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6  # class-specific context (False or True)
TRIALS=$7  # num of trials

for SEED in 1
do
    DIR=${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
#    MODEL_DIR=output/${DIR}
    MODEL_DIR=output/my_cifar100/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    OUTPUT_DIR=output/${TRAINER}/${CFG}_${SHOTS}shots/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}/${DATASET}/openness/base_prompt/

    python main.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml \
    --model-dir ${MODEL_DIR} \
    --target-label-dir ${OUTPUT_DIR} \
    --eval-only \
    --incremental-evaluation \
    --trials ${TRIALS} \
    --load-epoch 50 \
    TRAINER.COOP.N_CTX ${NCTX} \
    TRAINER.COOP.CSC ${CSC} \
    TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP}
done

# CUDA_VISIBLE_DEVICES=0 bash scripts/stability_extensibility_coop.sh my_cifar100 vit_b16_ep50 end 16 16 False 3
# CUDA_VISIBLE_DEVICES=0 bash scripts/stability_extensibility_coop.sh imagenet_entity13 vit_b16_ep50 end 16 16 False 3
# CUDA_VISIBLE_DEVICES=0 bash scripts/stability_extensibility_coop.sh imagenet_living17 vit_b16_ep50 end 16 16 False 3