import torch
import torch.nn as nn
from torch import Tensor


def initialize_mlp(input_sz:int, hidden_sz:int, output_sz:int, num_layers:int, slope:float, batch_norm:bool):
    """
    Initializes a Multi-Layer Perceptron (MLP) with specified parameters.

    Args:
        input_sz (int): Input size.
        hidden_sz (int): Size of hidden layers.
        output_sz (int): Output size.
        num_layers (int): Number of layers.
        slope (float): Slope parameter for LeakyReLU.
        batch_norm (bool): Whether to include batch normalization.

    Returns:
        nn.Sequential: A PyTorch MLP model.
    """
    activation_function = nn.LeakyReLU(slope)
    layers_list = [nn.Linear(input_sz, hidden_sz), activation_function]
    if batch_norm:
        layers_list.append(nn.BatchNorm1d(hidden_sz))

    for _ in range(num_layers - 2):
        layers_list.extend([
            nn.Linear(hidden_sz, hidden_sz),
            activation_function,
            nn.BatchNorm1d(hidden_sz) if batch_norm else None
        ])

    layers_list.append(nn.Linear(hidden_sz, output_sz))
    layers_list = [layer for layer in layers_list if layer is not None]

    model = nn.Sequential(*layers_list)

    # Initialize weights using Xavier initialization
    for layer in model:
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)

    return model


class SharedMLP(nn.Module):
    """
    Defines a Shared Multi-Layer Perceptron (MLP) with parameter sharing across inputs.

    Args:
        input_sz (int): Input size.
        hidden_sz (int): Hidden layer size.
        output_sz (int): Output size.
        layers (int): Number of layers.
        slope (float): Slope parameter for LeakyReLU.
        batch_norm (bool): Whether to include batch normalization.
    """
    def __init__(self, input_sz:int, hidden_sz:int, output_sz:int, num_layers:int, slope:float, batch_norm:bool):
        super(SharedMLP, self).__init__()
        activation_function = nn.LeakyReLU(slope) 

        self.common_layer = nn.ParameterList()  # Shared parameters for all layers
        self.phi_layers = nn.ModuleList()  # Layer-specific modules

        # Input layer
        self.common_layer.append(nn.Parameter(torch.empty(input_sz, hidden_sz)))
        self.phi_layers.append(nn.Sequential(
            nn.Linear(input_sz, hidden_sz),
            activation_function,
            nn.BatchNorm1d(hidden_sz) if batch_norm else None
        ))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.common_layer.append(nn.Parameter(torch.empty(hidden_sz, hidden_sz)))
            self.phi_layers.append(nn.Sequential(
                nn.Linear(hidden_sz, hidden_sz),
                activation_function,
                nn.BatchNorm1d(hidden_sz) if batch_norm else None
            ))

        # Output layer
        self.common_layer.append(nn.Parameter(torch.empty(hidden_sz, output_sz)))
        self.phi_layers.append(nn.Sequential(
            nn.Linear(hidden_sz, output_sz),
            activation_function
        ))

        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initializes parameters using Xavier initialization."""
        for param in self.common_layer:
            torch.nn.init.xavier_uniform_(param)

        for layer in self.phi_layers:
            for sub_layer in layer:
                if isinstance(sub_layer, nn.Linear):
                    torch.nn.init.xavier_uniform_(sub_layer.weight)

    def forward(self, input_1: Tensor, input_2: Tensor):
        """
        Forward pass with shared parameters across inputs.

        Args:
            input_1 (Tensor): First input tensor.
            input_2 (Tensor): Second input tensor.

        Returns:
            tuple: Transformed embeddings of input_1 and input_2.
        """
        for layer_idx, layer in enumerate(self.phi_layers):
            shared_term = (input_1.mean(0) + input_2.mean(0)) @ self.common_layer[layer_idx]
            input_1 = layer(input_1) + shared_term.view(1, -1)
            input_2 = layer(input_2) + shared_term.view(1, -1)
        return input_1.mean(0), input_2.mean(0)


class SharedProductNet(nn.Module):
    """
    Defines a Shared Product Network with encoder, phi, and rho components.

    Args:
        encoder_params (dict): Parameters for the encoder network.
        phi_params (dict): Parameters for the phi network.
        rho_params (dict): Parameters for the rho network.
        activation (str): Activation function name.
        bn (bool): Whether to use batch normalization.
        slope (float): Slope parameter for LeakyReLU.
    """
    def __init__(self, encoder_params: dict, phi_params: dict, rho_params: dict, bn: bool, slope: float):
        super(SharedProductNet, self).__init__()

        self.encoder = SharedMLP(
            input_sz=encoder_params['input_dim'],
            hidden_sz=encoder_params['hidden'],
            output_sz=encoder_params['output'],
            num_layers=encoder_params['layers'],
            batch_norm=bn,
            slope=slope)

        self.phi = initialize_mlp(
            input_sz=encoder_params['output'],
            hidden_sz=phi_params['hidden'],
            output_sz=phi_params['output'],
            num_layers=phi_params['layers'],
            batch_norm=False,
            slope=slope
        )

        self.rho = initialize_mlp(
            input_sz=phi_params['output'],
            hidden_sz=rho_params['hidden'],
            output_sz=1,
            num_layers=rho_params['layers'],
            batch_norm=False,
            slope=slope
        )

    def forward(self, input1: Tensor, input2: Tensor):
        """
        Forward pass for SharedProductNet.

        Args:
            input1 (Tensor): First input tensor.
            input2 (Tensor): Second input tensor.

        Returns:
            tuple: Model output and embeddings from phi network.
        """
        embd1, embd2 = self.encoder(input1, input2)
        embd1, embd2 = self.phi(embd1), self.phi(embd2)
        output = self.rho(embd1 + embd2)
        return output
