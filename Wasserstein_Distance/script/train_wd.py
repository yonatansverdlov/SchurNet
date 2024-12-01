import os
import numpy as np
import torch
import argparse
import yaml
from codes.fit_productnet import train_point_productnet, validation_loss
from codes.utils import WassersteinPairDataset, return_dim, return_path
from easydict import EasyDict


def return_dataset(dataset_name: str, small: bool):
    """
    Loads and processes training and validation datasets.

    Args:
        dataset_name (str): The name of the dataset.
        small (bool): If True, loads the 'small' version of the dataset.

    Returns:
        tuple: Training and validation datasets as WassersteinPairDataset objects.
    """
    # Retrieve dataset paths
    train_path, val_path = return_path(dataset_name=dataset_name, small=small)

    # Load datasets
    train_data = np.load(train_path, allow_pickle=True)
    val_data = np.load(val_path, allow_pickle=True)

    # Convert to WassersteinPairDataset format
    train_dataset = WassersteinPairDataset(
        train_data['Ps'], train_data['Qs'], torch.tensor(train_data['dists'])
    )
    val_dataset = WassersteinPairDataset(
        val_data['Ps'], val_data['Qs'], torch.tensor(val_data['dists'])
    )

    return train_dataset, val_dataset


def train_wd(dataset_name: str, seed: int = None, check_out_of_dist: bool = False):
    """
    Trains a SharedProductNet model and evaluates its performance.

    Args:
        dataset_name (str): Name of the dataset to use for training.
        seed (int, optional): Random seed for reproducibility. Defaults to the config seed.
        check_out_of_dist (bool, optional): If True, evaluates on out-of-distribution data.

    Returns:
        float: Validation loss on the small dataset.
    """
    # Get input dimension for the dataset
    dimension = return_dim(dataset_name=dataset_name)

    # Load configuration for the dataset
    with open('codes/config.yaml', 'r') as f:
        config = yaml.safe_load(f)[dataset_name]
    config = EasyDict(config)

    # Set random seed
    if seed is None:
        seed = config.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Load datasets
    train_dataset, val_dataset = return_dataset(dataset_name=dataset_name, small=True)

    # Model parameters
    embed_size = config.embed_size
    num_layer = config.num_layer
    mlp_params = {'hidden': embed_size, 'output': embed_size, 'layers': num_layer}
    phi_params = {'hidden': embed_size, 'output': embed_size, 'layers': num_layer}
    rho_params = {'hidden': embed_size, 'output': 1, 'layers': num_layer}

    # Training configuration
    max_iter = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Train the model
    shared_model = train_point_productnet(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        dimension=dimension,
        initial=mlp_params,
        phi=phi_params,
        rho=rho_params,
        device=device,
        lr=config.lr,
        iterations=max_iter,
        batch_size=config.batch_size,
        activation=config.act,
        batch=config.use_bn,
        wd=config.wd,
        factor=config.factor,
        opt_type=config.opt_type,
        slope=config.slope
    )

    # Validate the model
    val_loss_mean = validation_loss(val_dataset, shared_model, device)
    model_dir = f'../data/models/{dataset_name}/{config}'
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'Model_{val_loss_mean:.4f}.pth')
    torch.save(shared_model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

    # Evaluate on small dataset
    val_loss_small = validation_loss(val_dataset, shared_model, device)
    print(f"The validation loss on the small dataset is {val_loss_small:.4f}")

    # Evaluate on out-of-distribution dataset (if applicable)
    if check_out_of_dist:
        train_dataset, val_dataset = return_dataset(dataset_name=dataset_name, small=False)
        val_loss_gen = validation_loss(val_dataset, shared_model, device)
        print(f"The validation loss on the out-of-distribution dataset is {val_loss_gen:.4f}")

    return val_loss_small


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a SharedProductNet model.")
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='ncircle3',
        help='Name of the dataset to train on.'
    )
    args = parser.parse_args()

    # Train and evaluate the model across multiple seeds
    num_seeds = 20
    losses = []
    for seed in range(num_seeds):
        val_loss = train_wd(dataset_name=args.dataset_name, seed=seed)
        losses.append(val_loss)

    # Aggregate results
    losses = torch.tensor(losses)
    best_seed = torch.argmin(losses).item()
    best_loss = torch.min(losses).item()
    print(f"Best seed: {best_seed}, Best loss: {best_loss:.4f}")

