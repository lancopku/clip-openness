# Installation

This codebase is tested on Ubuntu 20.04.2 LTS with python 3.8. Follow the below steps to create environment and install dependencies.

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -n clip-openness python=3.8

# Activate the environment
conda activate clip-openness

# Install torch and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```

* Install dassl library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```

* Clone clip-openness code repository and install requirements
```bash
# Clone clip-openness code base
git clone https://github.com/lancopku/clip-openness.git

cd clip-openness/

# Install requirements
pip install -r requirements.txt
```