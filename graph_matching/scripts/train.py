"""
Training script for the graph model with multiple fixed seeds.
"""
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from easydict import EasyDict
import argparse
import yaml
from net.dataset import SyntheticGraphMatchingDataset
from net.lightning_model import LightningModel


def load_config(config_path: str = "config.yaml") -> EasyDict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        EasyDict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return EasyDict(config)


def set_seed(seed: int):
    """
    Set random seed for reproducibility.

    Args:
        seed (int): Seed value.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)


def train_and_evaluate(config, model_type, noise, seed):
    """
    Train and evaluate the model for a single seed.

    Args:
        config (EasyDict): Configuration dictionary.
        model_type (str): Model type.
        noise (float): Noise level.
        seed (int): Random seed.

    Returns:
        dict: Test results for this seed.
    """
    set_seed(seed)

    # Update config with specific parameters
    config.model_type = model_type
    config.noise = noise
    
    # Dataset and DataLoader
    train_dataset = SyntheticGraphMatchingDataset(n=20, num_graphs=10, noise=config.noise, num_samples=1000)
    val_dataset = SyntheticGraphMatchingDataset(n=20, num_graphs=10, noise=config.noise, num_samples=200)
    test_dataset = SyntheticGraphMatchingDataset(n=20, num_graphs=10, noise=config.noise, num_samples=200)

    train_loader = DataLoader(train_dataset, batch_size=config.bs, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.bs)
    test_loader = DataLoader(test_dataset, batch_size=config.bs)

    # Model
    model = LightningModel(config)

    # Callbacks: Save the best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", dirpath="data/checkpoints", filename="best_model-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1, mode="min", save_last=True
    )

    # Logger: Save logs to data/lightning_logs
    logger = CSVLogger("data", name=f"lightning_logs_{model_type}_noise_{noise}_seed_{seed}")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        callbacks=[checkpoint_callback],
        logger=logger,
        enable_checkpointing=True  # Avoid overwriting checkpoints for each seed
    )

    # Training
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Testing
    test_results = trainer.test(model, dataloaders=test_loader)
    return test_results[0]['test_acc']  # Return the first (and only) test result
                          
                        
if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Training script for graph models.")
    parser.add_argument("--model_type", type=str, required=True, help="Model type: SchurNet, Siamese, or DSS.")
    parser.add_argument("--noise", type=float, required=True, help="Noise level for the graphs.")
    args = parser.parse_args()

    # Load configuration
    config = load_config('./net/config.yaml')[args.model_type]

    # Collect results over all seeds
    all_results = []
    result = train_and_evaluate(config, args.model_type, args.noise, seed = config.seed)
    # Compute mean results

    # Print mean results
    print(f"Mean results for Model: {args.model_type}, Noise: {args.noise}")
    print(result)

