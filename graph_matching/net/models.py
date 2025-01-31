"""
The models.
"""

import torch
import torch.nn as nn
from torch import Tensor
from easydict import EasyDict
from net.decomposition import get_iso_matrix, batched_matrix_decomposition
from net.layer_utils import Dense


class SchurLayer(nn.Module):
    """
    Schur layer.
    """

    def __init__(self, input_dim: int, model_type: str, sigma='relu'):
        """
        Initialize the Schur layer.
        Args:
            input_dim (int): The input dimension.
            model_type (str): The model type (e.g., SchurNet, DSS).
            sigma (str): The activation function.
        """
        super().__init__()
        self.model_type = model_type
        self.iso_matrix = nn.Parameter(get_iso_matrix(), requires_grad=False)
        self.input_dim = input_dim

        # Activation function
        self.sigma = getattr(torch.nn, sigma.capitalize(), torch.nn.ReLU)()

        # Learnable parameters
        self.weight = nn.Parameter(torch.empty(9, 9))
        self.alpha = nn.Parameter(torch.empty(1, 1))
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        nn.init.xavier_uniform_(self.weight)
        nn.init.ones_(self.alpha)

    def forward(self, X: Tensor) -> Tensor:
        """
        Forward pass.
        Args:
            X (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        # Irreducible representations
        composed_parts = batched_matrix_decomposition(X, model_type=self.model_type)
        weighted_sum = (composed_parts @ (self.weight * self.iso_matrix)).sum(-1).permute(0, 1, 4, 3, 2)

        if self.model_type == 'DSS':
            X_sum = X.mean(1).mean(-1).mean(-1)
            batch, f = X_sum.size()
            X_sum = X_sum.view(batch,1,1,1,f)
            weighted_sum = weighted_sum + self.alpha * X_sum

        return weighted_sum


class GModels(nn.Module):
    """
    Graph model.
    """

    def __init__(self, config: EasyDict):
        """
        Initialize the graph model.
        Args:
            config (EasyDict): Configuration.
        """
        super().__init__()
        self.dims = config.dims
        self.model_type = config.model_type

        assert self.model_type in ['SchurNet', 'Siamese', 'DSS'], "Invalid model type!"

        layers = []
        for i in range(len(self.dims) - 1):
            layers.append(SchurLayer(input_dim=self.dims[i], model_type=self.model_type))
            layers.append(Dense(residual=False, seed=0, in_features=self.dims[i], out_features=self.dims[i + 1], activation_fn=nn.ReLU()))
        self.layers = nn.Sequential(*layers)

    def forward(self, A: Tensor) -> Tensor:
        """
        Forward pass through the model.
        Args:
            A (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.layers(A).squeeze().sum(dim=(-1, -2))
