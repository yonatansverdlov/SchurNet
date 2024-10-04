# Wessarstein Distance Computation

## Overview
This folder contains the code for Wessarstein Distance computation, which is part of the larger project. To install the necessary dependencies for this component, create the environment using the provided environment.yml file by running the following commands:

```bash
conda env create -f dependencies.yml
conda activate Wasserstein-env
```
## Data
Please download the datasets from https://drive.filen.io/f/69d1d525-1ce8-4770-88d6-a2cbc700785c#SXRTYQFcSUmGEirL8GQWZPEpSAaAY8EX.
## Training
Choose one of the datasets of ncircle3, ncircle6, random, mn_small(modelnet small), mn_large(modelnet large), rna(RNA) named set_name.
After downloading the data, please run the following command to start the training process:
```bash
python train_wd.py --dataset_name set_name
```
## Deep Weight Space Alignment
## Installation
```bash
conda create --name deep-align python=3.9
conda activate deep-align
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

cd DWS
pip install -e .
```

## MNIST MLP
To run the MLP experiments, first download the data:
```bash
mkdir datasets
wget "https://www.dropbox.com/s/sv85hrjswaspok4/mnist_classifiers.zip"
unzip -q mnist_classifiers.zip -d datasets
```
## CIFAR10 MLP
To run the MLP experiments, first download the data:
```bash
mkdir datasets
wget "https://www.dropbox.com/s/sv85hrjswaspok4/cifar_classifiers.zip"
unzip -q cifar_classifiers.zip -d datasets
```
## Split data:
```bash
Choose set_type in mnist, cifar 
python experiments/utils/data/generate_splits.py --set_type set_type
```
## Training
Choose set_type mnist or cifar
```bash
cd experiments/mlp_image_classifier
python trainer.py --set_type
```




