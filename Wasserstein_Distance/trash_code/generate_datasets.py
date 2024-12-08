import argparse
import numpy as np
import os
from codes.dataset import noisy_circles, fixed_point_set, build_dataset, build_comprehensive_sampler, load_hdf5_data

# Set random seed for reproducibility
np.random.seed(0)


def create_dataset(dataset_name: str, is_small: bool):
    """
    Creates training and validation datasets for a given dataset name.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'ncircle3', 'random', 'rna').
        is_small (bool): Indicates whether to create a 'small' or 'large' dataset.
    """
    # Define dataset-specific parameters
    dataset_params = {
        'ncircle3': {'train_sz': 2000, 'val_sz': 200, 'dim': 3},
        'ncircle6': {'train_sz': 3600, 'val_sz': 400, 'dim': 6},
        'random': {'train_sz': 3000, 'val_sz': 300, 'dim': 2},
        'mn_small': {'train_sz': 3000, 'val_sz': 300, 'dim': 3},
        'mn_large': {'train_sz': 4000, 'val_sz': 400, 'dim': 3},
        'rna': {'train_sz': 3000, 'val_sz': 300, 'dim': 2000}
    }

    if dataset_name not in dataset_params:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    params = dataset_params[dataset_name]
    train_sz = params['train_sz']
    val_sz = params['val_sz']
    dim = params['dim']

    nmax = 300 if is_small else 500
    nmin = 100 if is_small else 300

    if dataset_name in ['mn_large']:
        nmax = 2049 if is_small else 2000
        nmin = 2048 if is_small else 1800
    elif dataset_name == 'random':
        nmax = 257 if is_small else 300
        nmin = 256 if is_small else 200

    # Generate datasets based on dataset type
    if dataset_name in ['random', 'ncircle3', 'ncircle6']:
        if 'ncircle' in dataset_name:
            Ps, Qs, dists = noisy_circles(nmin=nmin, nmax=nmax, pairs=train_sz, dim=dim, order=2)
            Ps_val, Qs_val, dists_val = noisy_circles(nmin=nmin, nmax=nmax, pairs=val_sz, dim=dim, order=2)
        else:
            pointset = fixed_point_set(dim=dim, num=10000, data_type=dataset_name)
            Ps, Qs, dists = build_dataset(pointset, nmin=nmin, nmax=nmax, pairs=train_sz)
            Ps_val, Qs_val, dists_val = build_dataset(pointset, nmin=nmin, nmax=nmax, pairs=val_sz)

    elif dataset_name in ['mn_small', 'mn_large']:
        raw_data, labels, label_dict = load_hdf5_data('../data/samples/raw/modelnet.h5')
        Ps, Qs, dists = build_comprehensive_sampler(
            raw_data, label_dict, nmin=nmin, nmax=nmax, pairs=train_sz + val_sz, order=2
        )
        Ps_val = Ps[train_sz: train_sz + val_sz]
        Qs_val = Qs[train_sz: train_sz + val_sz]
        dists_val = dists[train_sz: train_sz + val_sz]

    elif dataset_name == 'rna':
        pointset = np.load('../data/samples/raw/rna.npy')
        Ps, Qs, dists = build_dataset(pointset, nmin=nmin, nmax=nmax, pairs=train_sz, order=2)
        Ps_val, Qs_val, dists_val = build_dataset(pointset, nmin=nmin, nmax=nmax, pairs=val_sz, order=2)

    # Create directories if needed and save datasets
    dataset_dir = f'../data/samples/{dataset_name}'
    os.makedirs(dataset_dir, exist_ok=True)

    size_label = 'small' if is_small else 'large'
    train_file = f'{dataset_dir}/train_{size_label}.npz'
    val_file = f'{dataset_dir}/val_{size_label}.npz'

    np.savez(train_file, Ps=Ps[:train_sz], Qs=Qs[:train_sz], dists=dists[:train_sz])
    np.savez(val_file, Ps=Ps_val, Qs=Qs_val, dists=dists_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Construct datasets')
    parser.add_argument('--dataset_name', type=str, required=True, help="Name of the dataset to construct")
    args = parser.parse_args()

    create_dataset(dataset_name=args.dataset_name, is_small=True)
    create_dataset(dataset_name=args.dataset_name, is_small=False)




