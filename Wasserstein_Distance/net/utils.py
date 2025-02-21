from torch.utils.data import Dataset
import torch

def return_path(dataset_name:str, small:bool=True):
    """
    Returns the file paths for the training and validation datasets.

    Args:
        dataset_name (str): Name of the dataset.
        small (bool): Indicates whether to use the 'small' or 'large' version of the dataset.

    Returns:
        tuple: Paths to the training and validation dataset files.
    """
    size_suffix = 'small' if small else 'large'
    train_path = f'./data/samples/{dataset_name}/train_{size_suffix}.npz'
    val_path = f'./data/samples/{dataset_name}/val_{size_suffix}.npz'
    return train_path, val_path


def return_dim(dataset_name:str):
    """
    Returns the input dimension of the dataset based on its name.

    Args:
        dataset_name (str): Name of the dataset.

    Returns:
        int: Input dimension of the dataset. None if the dataset is not found.
    """
    dataset_mapping = {
        "ncircle3": 3,
        "ncircle6": 6,
        "random": 2,
        "mn_small": 3,
        "mn_large": 3,
        "rna": 2000,
    }
    return dataset_mapping.get(dataset_name, None)

class WassersteinPairDataset(Dataset):
    def __init__(self, sources:list, targets:list, emds:list, device:str):
        self.sources = [p.to(dtype=torch.float32, device=device) if isinstance(p, torch.Tensor) else torch.tensor(p, dtype=torch.float32, device=device) for p in sources]
        self.targets = [p.to(dtype=torch.float32, device=device) if isinstance(p, torch.Tensor) else torch.tensor(p, dtype=torch.float32, device=device) for p in targets]
        self.emds = [p.to(dtype=torch.float32, device=device) if isinstance(p, torch.Tensor) else torch.tensor(p, dtype=torch.float32, device=device) for p in emds]
        self.device = device

    def __len__(self):
        return len(self.emds)

    def __getitem__(self, idx:int):
        source = self.sources[idx]
        target = self.targets[idx]
        emd = self.emds[idx]
        return source, target, emd


