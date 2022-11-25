### DeCLIP
```bash
mkdir -p checkpoints/declip
cd checkpoints/declip
# download the pre-trained ckpts from https://github.com/Sense-GVT/DeCLIP#our-pretrain-declip-model-w-text-encoder
# download the vocab file from https://github.com/Sense-GVT/DeCLIP/blob/main/docs/dataset_prepare.md#text
```

### SLIP
```bash
mkdir -p checkpoints/slip
cd checkpoints/slip
# download the pre-trained ckpts from https://github.com/facebookresearch/SLIP#vit-base
wget https://dl.fbaipublicfiles.com/slip/slip_base_100ep.pt
# download the vocab file from https://github.com/Sense-GVT/DeCLIP/blob/main/docs/dataset_prepare.md#text
```

### CoOp
Download the pre-trained weights of CoOp (both M=16 & M=4) on ImageNet based on RN50, RN101, ViT-B/16 and ViT-B/32 according to [link](https://github.com/KaiyangZhou/CoOp#models-and-results).
The directory structure should look like:
```
output/
|–– imagenet/
|   |–– CoOp/
|       |–– rn101_ep50_16shots/
|       |–– rn50_ep50_16shots/
|       |–– vit_b16_ep50_16shots/
|       |–– vit_b32_ep50_16shots/
```