# 1. Extensibility-Stability Evaluation

## (1) Intra-dataset vocab expansion

Below, we provide instructions to evaluate extensibility and stability for CLIP-like models.

```bash
# Evaluate for CIFAR100
CUDA_VISIBLE_DEVICES=0 bash scripts/stability_extensibility.sh ZeroshotCLIP my_cifar100 vit_b32 3

# Evaluate for ImageNet-Entity13
CUDA_VISIBLE_DEVICES=1 bash scripts/stability_extensibility.sh ZeroshotCLIP imagenet_entity13 vit_b32 3

# Evaluate for ImageNet-Living17
CUDA_VISIBLE_DEVICES=2 bash scripts/stability_extensibility.sh ZeroshotCLIP imagenet_living17 vit_b32 3
```

## (2) Dataset-level vocab expansion
```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/stability_extensibility.sh ZeroshotCLIP flowers_pets_cars vit_b32 2

CUDA_VISIBLE_DEVICES=1 bash scripts/stability_extensibility.sh ZeroshotCLIP cifar100_caltech101_sun397 vit_b32 2

CUDA_VISIBLE_DEVICES=2 bash scripts/stability_extensibility.sh ZeroshotCLIP cifar10_cifar100_imagenet vit_b32 2
```

## (3) Adversarial vocab mining
TBD

## (4) Vocab expansion for coop
```bash
mkdir -p output/my_cifar100/CoOp/vit_b16_ep50_16shots/nctx16_cscFalse_ctpend/seed1
CUDA_VISIBLE_DEVICES=0 bash scripts/stability_extensibility_coop.sh my_cifar100 vit_b16_ep50 end 16 16 False 3
```

# 2. Intra-modal Uniformity and Inter-modal Alignment
TBD

# 3. Retrieval-enhanced prompt engineering (REPE)

## (1) Retrieve image-caption pairs based on [clip-retrieval](https://github.com/rom1504/clip-retrieval)
Download pre-retrieved image-caption pairs from LAION-5B:
- For CIFAR100: [link](https://github.com/lancopku/clip-openness/releases/download/v0.1.0/laion5B_retrieval_1000_for_cifar100.zip)
- For ImageNet: [link](https://github.com/lancopku/clip-openness/releases/download/v0.1.0/laion5B_retrieval_1000_for_imagenet.zip)

The directory structure should look like:
```
data/
|–– cifar100/
|   |–– laion5B_retrieval_1000/
|       |–– ... # a bunch of .json files
|–– imagenet/
|   |–– laion5B_retrieval_1000/
|       |–– ... # a bunch of .json files
```

You can also retrieve image-caption pairs for other downstream datasets or backbones followed by [retrieval.py](../repe/retrieval.py), [download_photo.py](../repe/download_photo.py) and [dump_features.py](../repe/dump_features.py)

```bash
python repe/retrieval.py
```

## (2) REPE for extensibility-stability evaluation
```bash
# Evaluate for CIFAR100
CUDA_VISIBLE_DEVICES=0 bash scripts/stability_extensibility_repe.sh ZeroshotCLIP2 my_cifar100 vit_b32 0.25 100 3

# Evaluate for ImageNet-Entity13
CUDA_VISIBLE_DEVICES=1 bash scripts/stability_extensibility_repe.sh ZeroshotCLIP2 imagenet_entity13 vit_b32 0.25 100 3

# Evaluate for ImageNet-Living17
CUDA_VISIBLE_DEVICES=2 bash scripts/stability_extensibility_repe.sh ZeroshotCLIP2 imagenet_living17 vit_b32 0.25 100 3
```

## (3) REPE for standard zero-shot classification (CIFAR100)
```bash
CUDA_VISIBLE_DEVICES=0 sh scripts/zeroshot_repe.sh ZeroshotCLIP2 my_cifar100 vit_b32 0.25 100
```