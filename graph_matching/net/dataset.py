"""
Synthetic dataset for graph matching tasks.
"""

import torch
from torch.utils.data import Dataset


def generate_bernoulli_graph_torch(n: int, p: float = 0.5) -> torch.Tensor:
    return torch.bernoulli(torch.full((n, n), p))


def permute_graph_torch(graph: torch.Tensor) -> torch.Tensor:
    perm = torch.randperm(graph.size(0))
    return graph[perm][:, perm]

def add_noise_to_graph(graph: torch.Tensor, noise_level: float) -> torch.Tensor:
    """
    Add Gaussian noise pairwise to the adjacency matrix with a specified noise level.

    Args:
        graph (torch.Tensor): Adjacency matrix of the graph.
        noise_level (float): Noise level indicating the standard deviation of the noise.

    Returns:
        torch.Tensor: Noisy adjacency matrix.
    """
    # Generate element-wise Gaussian noise
    random_noise = torch.randn_like(graph) * noise_level  # Each element gets its own noise
    
    # Add the noise to the graph
    noisy_graph = graph + random_noise
    
    return noisy_graph

def generate_dataset_torch(n: int, k: int, noise_level: float) -> torch.Tensor:
    graph1 = generate_bernoulli_graph_torch(n)
    dataset = [graph1]

    for _ in range(k - 2):
        permuted_graph = permute_graph_torch(graph1)
        dataset.append(add_noise_to_graph(permuted_graph, noise_level))

    graph2 = add_noise_to_graph(generate_bernoulli_graph_torch(n), noise_level)
    random_index = torch.randint(0, k, (1,)).item()
    dataset.insert(random_index, graph2)

    return torch.stack(dataset).unsqueeze(1), random_index


class SyntheticGraphMatchingDataset(Dataset):
    def __init__(self, n: int, num_graphs: int, noise: float = 0.0, num_samples: int = 1000):
        self.n = n
        self.num_graphs = num_graphs
        self.noise = noise
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        torch.manual_seed(idx)
        return generate_dataset_torch(self.n, self.num_graphs, self.noise)
