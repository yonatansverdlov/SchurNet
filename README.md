# SchurNet

### Overview
This repo contains code for the experimental part of the paper ["Revisiting Multi-Permutation Equivariance Thourgh The Lens of Irreducible Representations"](https://arxiv.org/abs/2410.06665).

## Wasserstein Distance Computation
To install the necessary dependencies for this component, create the environment using the provided environment.yml file by running the following commands:

```bash
conda env create -f dependencies.yml
conda activate weis_dis
```
### Data
Download the datasets from [Here](https://drive.filen.io/f/69d1d525-1ce8-4770-88d6-a2cbc700785c#SXRTYQFcSUmGEirL8GQWZPEpSAaAY8EX).
Unzip the downloaded file and place the data in the `Wasserstein_Distance` folder.
### Training
Choose one of the datasets of ncircle3, ncircle6, random, mn_small(modelnet small), mn_large(modelnet large), rna(RNA) named set_name.
After downloading the data, please run the following command to start the training process:
```bash
cd Wasserstein_Distance/script
python train_wd.py --dataset_name set_name
```
## Deep Weight Space Alignment
### Installation
```bash
conda create --name deep-align python=3.9
conda activate deep-align
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia

cd DWS
pip install -r requirements.txt
```
### Data
Choose set_type to be `mnist` or `cifar`.

### MNIST MLP
To run the MLP experiments, first download the data:
```bash
mkdir -p datasets
wget "https://www.dropbox.com/s/sv85hrjswaspok4/mnist_classifiers.zip" -P data/datasets
unzip -q data/datasets/mnist_classifiers.zip -d data/datasets/samples

```
### CIFAR10 MLP
To run the MLP experiments, first download the data:
```bash
mkdir -p datasets
wget "https://www.dropbox.com/s/sv85hrjswaspok4/cifar_classifiers.zip" -P data/datasets
unzip -q data/datasets/cifar_classifiers.zip -d data/datasets/samples
```
### Split data:
Run
```bash
python code/experiments/utils/data_utils/generate_splits.py --set_type set_type
```
### Training
For our model with the shared layers run:
```bash
python code/experiments/mlp_image_classifier/trainer.py --set_type set_type --shared True
```
For the baseline of Siamese model, run:
```bash
python code/experiments/mlp_image_classifier/trainer.py --set_type set_type --shared False
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
