# SchurNet
### Overview
This repo contains code for the experimental part of the paper ["Revisiting Multi-Permutation Equivariance Thourgh The Lens of Irreducible Representations"](https://arxiv.org/abs/2410.06665).
## General Setup
To create a conda environment that is good for all experiments, please run 
```bash
cd SchurNet
conda env create -f dependencies.yml 
conda activate SchurNet
```
## Graph Matching
In this experiment, we highlight the need of all linear equivariant layers. 
Choose a `model_type` in `Siamese`, `SchurNet`, and `DSS`. Choose `noise_level` and run:
```bash
cd graph_matching
python train.py --model_type model_type --noise noise_level
```
## Wasserstein Distance Computation
### Data
Download the datasets from [Here](https://www.kaggle.com/datasets/yonatansverdlov/data-for-wasserstein-distance-computation).
Unzip the downloaded file and place the data in the `Wasserstein_Distance/data` folder.
### Training
Choose `set_name` in `ncircle3`, `ncircle6`, `random`, `mn_small`, `mn_large`, `rna`.
After downloading the data, please run the following command to start the training process:
```bash
cd Wasserstein_Distance
python train_wd.py --dataset_name set_name
```
## Deep Weight Space Alignment
```bash
cd DWS
```
### Data
Choose `data_name` to be `mnist` or `cifar`.

### MNIST MLP
To run the MLP experiments, first download the data:
```bash
mkdir -p data
wget "https://www.dropbox.com/s/sv85hrjswaspok4/mnist_classifiers.zip" -P data
unzip -q data/mnist_classifiers.zip -d data/samples
python utils/data_utils/generate_splits.py --set_type mnist
```
### CIFAR10 MLP
To run the MLP experiments, first download the data:
```bash
mkdir -p data
wget "https://www.dropbox.com/scl/fi/lex7rj1147nhq2hsp83r1/cifar10_mlps.zip?rlkey=tiyq14zl70hjbmhq2y9sg14xo&dl=1" -P data/
unzip -q data/cifar_classifiers.zip -d data/samples
python utils/data_utils/generate_splits.py --set_type cifar
```
### Training
For our model with the shared layers run:
```bash
python trainer.py --data_name data_name --add_common True
```
For the baseline of Siamese model, run:
```bash
python trainer.py --data_name data_name --add_common False
```

## Citation
```
@misc{sverdlov2024revisiting,
      title={Revisiting Multi-Permutation Equivariance through the Lens of Irreducible Representations}, 
      author={Yonatan Sverdlov and Ido Springer and Nadav Dym},
      year={2024},
      eprint={2410.06665},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.06665}, 
}
```
