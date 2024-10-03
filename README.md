# Wessarstein Distance Computation

## Overview
This folder contains the code for Wessarstein Distance computation, which is part of the larger project. To install the necessary dependencies for this component, create the environment using the provided environment.yml file by running the following commands:

```bash
conda env create -f environment.yml
conda activate Wasserstein-env
```
For the datasets ncircle3, ncircle6, and random, please run the following commands to prepare and process the data:
```bash
cd Wasserstein_Distance/script/
python generate_datasets.py --dataset_name name
```

For the datasets mn_small, mn_large, and rna, please download them from https://drive.filen.io/f/69d1d525-1ce8-4770-88d6-a2cbc700785c#SXRTYQFcSUmGEirL8GQWZPEpSAaAY8EX and place them in the raw folder. 
After downloading, run the following command:
```bash
python generate_datasets.py --dataset_name name
```
After downloading the data, please run the following command to start the training process:
```bash
python train_wd.py --dataset_name name
```


