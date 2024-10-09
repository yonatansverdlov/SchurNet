import argparse
import json
import itertools
import numpy as np
import random
from dataset import *
import os

np.random.seed(0)


def create_dataset(dataset_name: str, is_small: bool):
    if dataset_name == 'ncircle3':
        train_sz = 2000
        val_sz = 200
        nmax = 300 if is_small else 500
        nmin = 100 if is_small else 300
        dim = 3
    elif dataset_name == 'ncircle6':
        train_sz = 3600
        val_sz = 400
        nmax = 300 if is_small else 500
        nmin = 100 if is_small else 300
        dim = 6
    elif dataset_name == 'random':
        nmax = 257 if is_small else 300
        nmin = 256 if is_small else 200
        train_sz = 3000
        val_sz = 300
        dim = 2
    elif dataset_name == 'mn_small':
        nmin = 20 if is_small else 300
        nmax = 200 if is_small else 500
        train_sz = 3000
        val_sz = 300
        dim = 3
    elif dataset_name == 'mn_large':
        train_sz = 4000
        val_sz = 400
        nmax = 2049 if is_small else 2000
        nmin = 2048 if is_small else 1800
        dim = 3
    elif dataset_name == 'rna':
        nmin = 20 if is_small else 300
        nmax = 200 if is_small else 500
        train_sz = 3000
        val_sz = 300
        dim = 2000

    if dataset_name in ['random', 'ncircle3', 'ncircle6']:

        if 'ncircle' in dataset_name:
            Ps, Qs, dists = noisy_circles(nmin=nmin,
                                          nmax=nmax,
                                          pairs=train_sz,
                                          dim=dim,
                                          order=2)
            Ps_val, Qs_val, dists_val = noisy_circles(nmin=nmin,
                                                      nmax=nmax,
                                                      pairs=val_sz,
                                                      dim=dim,
                                                      order=2)
        else:
            n = 10000
            pointset = fixed_point_set(dim=dim, num=n, data_type=dataset_name)
            Ps, Qs, dists = build_dataset(pointset,
                                          nmin=nmin,
                                          nmax=nmax,
                                          pairs=train_sz)
            Ps_val, Qs_val, dists_val = build_dataset(pointset,
                                                      nmin=nmin,
                                                      nmax=nmax,
                                                      pairs=val_sz)

    elif dataset_name in ['mn_small', 'mn_large']:
        # Loading raw ModelNet data
        raw_data, labels, label_dict = load_hdf5_data('../data/samples/raw/modelnet.h5')
        # build train dataset
        Ps, Qs, dists = build_comprehensive_sampler(raw_data,
                                                    label_dict,
                                                    nmin=nmin,
                                                    nmax=nmax,
                                                    pairs=train_sz + val_sz,
                                                    order=2)
        print(len(Ps), len(Qs))
        Ps_val, Qs_val, dists_val = Ps[train_sz: train_sz + val_sz], Qs[train_sz: train_sz + val_sz], dists[
                                                                                                      train_sz: train_sz + val_sz]


    elif dataset_name == 'rna':
        pointset = np.load('../data/samples/raw/rna.npy')
        Ps, Qs, dists = build_dataset(pointset,
                                      nmin=nmin,
                                      nmax=nmax,
                                      pairs=train_sz,
                                      order=2)
        Ps_val, Qs_val, dists_val = build_dataset(pointset,
                                                  nmin=nmin,
                                                  nmax=nmax,
                                                  pairs=val_sz,
                                                  order=2)

    top_lvl = '../data/samples/{}'.format(dataset_name)
    if not os.path.exists(top_lvl):
        os.makedirs(top_lvl)
    train_sf = '../data/samples/{}/train_{}.npz'.format(dataset_name, 'small' if is_small else 'lrage')
    val_sf = '../data/samples/{}/val_{}.npz'.format(dataset_name, 'small' if is_small else 'lrage')
    np.savez(train_sf, Ps=Ps[:train_sz], Qs=Qs[:train_sz], dists=dists[:train_sz])
    np.savez(val_sf, Ps=Ps_val, Qs=Qs_val, dists=dists_val)


parser = argparse.ArgumentParser(description='construct datasets')
parser.add_argument('--dataset_name', type=str)

arg = parser.parse_args()

create_dataset(dataset_name=arg.dataset_name, is_small=True)

create_dataset(dataset_name=arg.dataset_name, is_small=False)
