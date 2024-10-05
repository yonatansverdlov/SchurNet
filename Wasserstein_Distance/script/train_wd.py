# todo
# train and get similar results as in the paper
# add our shared layers
# train again and check results
import os
import numpy as np
import torch
import argparse
import yaml
import torch.nn as nn
from models import SharedProductNet
from fit_productnet import train_point_productnet, validation_loss
import itertools
from easydict import EasyDict
from utils import SetTypes, EMDPairDataset

def return_dataset(set_type,small ):
    train_sf, val_sf = set_type.return_path(small=small)                                                            
    train_data = np.load(train_sf, allow_pickle=True)
    Ps = train_data['Ps']
    Qs = train_data['Qs']
    dists = train_data['dists']
    val_data = np.load(val_sf, allow_pickle=True)
    Ps_val = val_data['Ps']
    Qs_val = val_data['Qs']
    dists_val = val_data['dists']
    # Create dataset
    train_dataset = EMDPairDataset(Ps, Qs, torch.tensor(dists))
    val_dataset = EMDPairDataset(Ps_val, Qs_val, dists_val)
    return train_dataset, val_dataset

def train_wd(set_type):
    if set_type == 'ncircle3':
        set_type = SetTypes.ncircle3
    elif set_type == 'ncircle6':
        set_type = SetTypes.ncircle6
    elif set_type == 'random':
        set_type = SetTypes.random
    elif set_type == 'mn_small':
        set_type = SetTypes.mn_small
    elif set_type == 'mn_large':
        set_type = SetTypes.mn_large
    elif set_type == 'rna':
        set_type = SetTypes.rna
    dimension = set_type.return_dim()
    dataset_name = set_type.return_dataset_name()

    with open('config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)[dataset_name]
        config = EasyDict(config)
    seed = config.seed
    torch.manual_seed(seed)

    # If using CUDA, set the seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 

    train_dataset, val_dataset = return_dataset(set_type=set_type,small = True)
    # now train
    embed_size = config.embed_size
    num_layer = config.num_layer
    mlp_params = {'hidden': embed_size, 'output': embed_size, 'layers':num_layer}
    phi_params = {'hidden': embed_size, 'output': embed_size, 'layers': num_layer}
    rho_params = {'hidden': embed_size, 'output': 1, 'layers':num_layer}
    embedding_size = phi_params['output']
    modelname = 'try_shared'
    max_iter = 40
    device = 'cuda'
    factor = config.factor
    lr = config.lr
    wd = config.wd
    size = config.batch_size
    use_bn = config.use_bn
    opt_type = config.opt_type
    act = config.act
    slope = config.slope

    shared_model, epochs_trained = train_point_productnet(train_dataset= train_dataset,
                                                val_dataset= val_dataset,
                                                dimension= dimension,
                                                initial= mlp_params,
                                                phi = phi_params,
                                                rho = rho_params,
                                                device=device,
                                                lr=lr,
                                                name=modelname,
                                                iterations=max_iter,
                                                batch_size=size,                              
                                                activation=act, 
                                                batch=use_bn,
                                                wd = wd,
                                                factor=factor,
                                                opt_type=opt_type,
                                                slope=slope)
    val_loss_mean, val_loss_std = validation_loss(val_dataset, shared_model, device, image=False)
    path_to_model =f'../data/models/{dataset_name}/{config}'
    if not os.path.exists(path_to_model):
        os.makedirs(path_to_model)
    path_to_model = os.path.join(path_to_model,f'Model_{val_loss_mean}')
    torch.save(shared_model.state_dict(),path_to_model)
    val_loss,_ = validation_loss(model=shared_model, val_dataset=val_dataset, device='cuda')
    print(f"The test loss on the small disterbution is {val_loss.item()}")
    # Check generalization.
    train_dataset, val_dataset = return_dataset(set_type=set_type,small = False)
    val_loss_gen,_ = validation_loss(model=shared_model, val_dataset=val_dataset, device='cuda')
    print(f"The test loss on the out of disterbution is {val_loss_gen.item()}")
    print(config)

parser = argparse.ArgumentParser(description="Process dataset with a specified radius.")

# Add dataset_name as a string argument (positional)
parser.add_argument('--dataset_name', type=str, help='Name of the dataset')

# Parse the arguments
args = parser.parse_args()

train_wd(set_type=args.dataset_name)