from enum import Enum
from functools import partial
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class SetTypes(Enum):
    ncircle3 = 0 
    ncircle6 = 1
    random = 2
    mn_small = 3
    mn_large = 4
    rna = 5

    def return_path(self, small = True):
        dataset_name = self.return_dataset_name()
        train_sf = '../data/samples/{}/train_{}.npz'.format(dataset_name,'small' if small else 'lrage' )
        val_sf = '../data/samples/{}/val_{}.npz'.format(dataset_name,'small' if small else 'lrage' )
        return train_sf, val_sf

    def return_dim(self):
        if self is SetTypes.ncircle3:
            return 3
        elif self is SetTypes.ncircle6:
            return 6
        elif self is SetTypes.random:
            return 2
        elif self is SetTypes.mn_small or self is SetTypes.mn_large:
            return 3
        elif self is SetTypes.rna:
            return 2000

    
    def return_dataset_name(self):
        if self is SetTypes.ncircle3:
            return f'ncircle3'
        if self is SetTypes.ncircle6:
            return f'ncircle6'   
        if self is SetTypes.random:
            return 'random'
        if self is SetTypes.mn_small:
            return 'mn_small'
        if self is SetTypes.mn_large:
            return 'mn_large'
        if self is SetTypes.rna:
            return 'rna'

def return_act(act_name:str,slope:float):
    if act_name == 'relu':
        func = nn.ReLU
    elif act_name == 'lrelu':
        func =  partial(nn.LeakyReLU,slope)
    elif act_name == 'tanh':
        func = nn.Tanh
    return func


class EMDPairDataset(Dataset):
    def __init__(self, sources, targets, emds):
        self.sources = sources
        self.targets = targets
        self.emds = emds

    def __len__(self):
        return len(self.emds)

    def __getitem__(self, idx):
        return self.sources[idx], self.targets[idx], self.emds[idx]

    def shuffle(self):
        permutation = np.random.permutation(len(self.emds))
        self.sources = self.sources[permutation]
        self.targets = self.sources[permutation]