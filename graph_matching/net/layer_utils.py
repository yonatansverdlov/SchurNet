import torch
from torch import Tensor
import torch.nn as nn


def _standardize(weight: Tensor) -> Tensor:
    """
    Normalize a weight tensor to ensure variance = 1 and mean = 0.

    Args:
        weight (Tensor): The input weight tensor.

    Returns:
        Tensor: Normalized weight tensor.
    """
    eps = 1e-6  # Small constant to prevent division by zero

    if len(weight.shape) == 3:
        axis = [0, 1]  # Normalize along all dimensions except the output dimension
    else:
        axis = 1  # For standard 2D weights

    var, mean = torch.var_mean(weight, dim=axis, keepdim=True)
    normalized_weight = (weight - mean) / (var + eps) ** 0.5
    return normalized_weight


def he_orthogonal_init(weight: Tensor, seed: int) -> Tensor:
    """
    Initialize a weight tensor using He orthogonal initialization.

    Args:
        weight (Tensor): The weight tensor to initialize.
        seed (int): Random seed for reproducibility.

    Returns:
        Tensor: He-orthogonally initialized weight tensor.
    """
    torch.manual_seed(seed)
    tensor = torch.nn.init.orthogonal_(weight)

    if len(tensor.shape) == 3:  # For 3D weights
        fan_in = tensor.shape[:-1].numel()
    else:  # For 2D weights
        fan_in = tensor.shape[1]

    with torch.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5

    return tensor


class Dense(nn.Module):
    """
    Dense (fully connected) layer with optional residual connection and custom initialization.
    """

    def __init__(
        self,
        seed: int,
        in_features: int,
        out_features: int,
        bias: bool = True,
        residual: bool = True,
        activation_fn: nn.Module = nn.Identity(),
    ):
        """
        Initializes the Dense layer.

        Args:
            seed (int): Random seed for initialization.
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): Whether to include a bias term. Default is True.
            residual (bool): Whether to add a residual connection. Default is True.
            activation_fn (torch.nn.Module): Activation function to use. Default is nn.Identity().
        """
        super().__init__()
        assert activation_fn is not None, "Activation function cannot be None"
        
        self.seed = seed
        self.residual = residual
        self.in_features = in_features
        
        # Initialize the linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()
        
        self.weight = self.linear.weight
        self.bias = self.linear.bias
        self.activation = activation_fn

    def reset_parameters(self):
        """
        Resets the parameters of the Dense layer using He orthogonal initialization.
        """
        if self.in_features != 1:  # Skip initialization for 1D input
            he_orthogonal_init(self.linear.weight, seed=self.seed)
        if self.linear.bias is not None:
            self.linear.bias.data.fill_(0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the Dense layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, ..., in_features).

        Returns:
            Tensor: Output tensor of shape (batch_size, ..., out_features).
        """
        x = self.linear(x)
        if self.residual:
            x = self.activation(x) + x
        else:
            x = self.activation(x)
        return x.transpose(2, -1)  # Ensure consistent output shape
