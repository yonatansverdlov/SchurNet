# Wessarstein Distance Computation

## Overview
This folder contains the code for Wessarstein Distance computation, which is part of the larger project. To install the necessary dependencies for this component, create the environment using the provided environment.yml file by running the following commands:

```bash
conda env create -f environment.yml
conda activate Wasserstein-env
```
## Data
Please download the datasets from https://drive.filen.io/f/69d1d525-1ce8-4770-88d6-a2cbc700785c#SXRTYQFcSUmGEirL8GQWZPEpSAaAY8EX .
## Training
Choose one of the datasets of ncircle3, ncircle6, random, mn_small(modelnet small), mn_large(modelnet large), rna named set_name.
After downloading the data, please run the following command to start the training process:
```bash
python train_wd.py --dataset_name set_name
```


