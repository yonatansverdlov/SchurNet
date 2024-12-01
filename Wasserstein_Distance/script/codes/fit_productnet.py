import numpy as np
import torch
import copy
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from codes.models import SharedProductNet
from tqdm import trange

# Define the Mean Squared Error (MSE) loss
MSELOSS = nn.MSELoss(reduction='mean')


def train_point_productnet(
    train_dataset: Dataset,
    val_dataset: Dataset,
    dimension: int,
    initial: dict,
    phi: dict,
    rho: dict,
    device: str,
    lr: float,
    factor: float,
    slope: float,
    activation: str = 'relu',
    iterations: int = 200,
    batch_size: int = 64,
    batch: bool = True,
    wd: float = 0.0,
    opt_type: str = 'Adam'
):
    """
    Trains the SharedProductNet model on the given dataset.

    Args:
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        dimension (int): Input dimensionality of the data.
        initial (dict): Parameters for the initial layers of the model.
        phi (dict): Parameters for the phi module.
        rho (dict): Parameters for the rho module.
        device (str): Device to run the model on ('cuda' or 'cpu').
        lr (float): Learning rate for the optimizer.
        factor (float): Factor for learning rate reduction.
        slope (float): Slope for LeakyReLU activation.
        activation (str): Activation function ('relu', 'lrelu', 'tanh'). Default is 'relu'.
        iterations (int): Number of training iterations. Default is 200.
        batch_size (int): Batch size for training. Default is 64.
        batch (bool): Whether to use batch normalization. Default is True.
        wd (float): Weight decay for the optimizer. Default is 0.0.
        opt_type (str): Optimizer type ('Adam' or 'AdamW'). Default is 'Adam'.

    Returns:
        SharedProductNet: Best trained model with the lowest validation loss.
    """
    best_loss = np.inf
    best_model = None

    # Update the input dimension for the model
    initial['input_dim'] = dimension
    model = SharedProductNet(initial, phi, rho, activation=activation, bn=batch, slope=slope).to(device)

    # Choose optimizer
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd) if opt_type == 'Adam' else AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=factor, patience=1)

    # Training loop
    for epoch in trange(iterations, desc="Training Progress"):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0

        # Batch training
        for i in trange(len(train_dataset), desc=f"Epoch {epoch + 1}"):
            input1 = train_dataset[i][0].float().to(device)
            input2 = train_dataset[i][1].float().to(device)
            yval = train_dataset[i][2].float().to(device)

            # Forward pass and loss computation
            pred, _, _ = model(input1, input2)
            yval = torch.unsqueeze(yval, dim=0)
            loss = MSELOSS(pred, yval) / batch_size
            epoch_loss += loss.item()

            # Backward pass
            loss.backward()
            if (i + 1) % batch_size == 0 or i == len(train_dataset) - 1:
                optimizer.step()
                optimizer.zero_grad()

        # Validation loss and scheduler step
        val_loss, _ = validation_loss(val_dataset=val_dataset, model=model, device=device)
        scheduler.step(val_loss)

        # Save the best model
        if val_loss < best_loss:
            best_model = copy.deepcopy(model)
            best_loss = val_loss

        print(f"Validation Loss at Epoch {epoch + 1}: {val_loss}")

    return best_model


def validation_loss(val_dataset: Dataset, model: SharedProductNet, device: str, image: bool = False):
    """
    Computes the validation loss for the given dataset.

    Args:
        val_dataset (Dataset): Validation dataset.
        model (SharedProductNet): The trained model.
        device (str): Device to run the model on ('cuda' or 'cpu').
        image (bool): Whether the input data is an image. Default is False.

    Returns:
       float: Mean validation loss.
    """
    total_loss = []
    with torch.no_grad():
        for i in range(len(val_dataset)):
            input1 = val_dataset[i][0].to(device)
            input2 = val_dataset[i][1].to(device)
            yval = torch.tensor(val_dataset[i][2]).to(device)

            # Forward pass
            pred, _, _ = model(input1, input2)

            # Loss computation (avoiding division by zero)
            if yval > 0.0:
                loss = torch.abs(pred - yval) / yval
                total_loss.append(loss.item())

    return np.mean(total_loss)


