"""
Decomposition of batched multi-featured graphs into irreducible representations (irreps).
"""

import torch
from torch import Tensor
from typing import List


def batched_matrix_decomposition(A: Tensor, model_type: str) -> Tensor:
    """
    Perform batched matrix decomposition for multi-featured graphs.

    Args:
        A (Tensor): A batch tensor of shape [batch, num_graphs, num_features, n, n].
        model_type (str): The model type ('SchurNet' or others) to determine decomposition behavior.

    Returns:
        Tensor: Decomposed tensors stacked along the last dimension.
    """
    batch_size, num_graphs, num_features, n, _ = A.shape
    device = A.device

    # Create diagonal matrix with all diagonal elements set to 1
    diagonal = torch.eye(n, device=device).repeat(batch_size, num_graphs, num_features, 1, 1)

    # Compute a0: Diagonal matrix with identical entries
    a0 = (A * diagonal).sum(dim=(-1, -2)).view(batch_size, num_graphs, num_features, 1, 1) * diagonal / n

    # Compute off-diagonal components
    off_diag = (A - a0).view(-1, num_graphs, num_features, n, n)
    off_diag_sum = off_diag.sum(dim=(-1, -2))
    a1 = (1 - diagonal) * off_diag_sum.view(-1, num_graphs, num_features, 1, 1) / (n**2 - n)

    # Remove components accounted for in a0 and a1
    A_hat = A - a0 - a1

    # Compute row and column sums
    row_sum = A_hat.sum(dim=-1)
    col_sum = A_hat.sum(dim=-2)

    # Extract diagonal of A_hat
    diag = A_hat.diagonal(dim1=-2, dim2=-1)

    # Compute duplicate rows and columns
    r = (n * diag - (n - 1) * row_sum - col_sum) / (n**2 - 2 * n)
    c = (n * diag - (n - 1) * col_sum - row_sum) / (n**2 - 2 * n)

    # Compute d for sum-zero diagonal matrix
    d = -(r + c + diag)

    # Decompose further into additional irreducible components
    a2 = -torch.diag_embed(d)  # Sum-zero diagonal matrix
    a3 = -c.unsqueeze(-2).expand(-1, -1, -1, n, -1)  # Duplicate columns
    a4 = -r.unsqueeze(-1).expand(-1, -1, -1, -1, n)  # Duplicate rows
    A_tilde = A_hat - a2 - a3 - a4

    # Compute anti-symmetric and symmetric row sum-zero matrices
    a5 = 0.5 * (A_tilde - A_tilde.transpose(-1, -2))
    a6 = 0.5 * (A_tilde + A_tilde.transpose(-1, -2))

    # Further compose the first two spaces based on model_type
    if model_type == 'SchurNet':
        b_0 = a0.mean(dim=1, keepdim=True).repeat(1, num_graphs, 1, 1, 1)
        b_1 = a0 - b_0
        b_2 = a1.mean(dim=1, keepdim=True).repeat(1, num_graphs, 1, 1, 1)
        b_3 = a1 - b_2
    else:
        b_0 = a0
        b_1 = torch.zeros_like(a0)
        b_2 = a1
        b_3 = torch.zeros_like(a0)

    # Stack all components along the last dimension
    return torch.stack([b_0, b_1, b_2, b_3, a2, a3, a4, a5, a6], dim=-1)


def fill_indices(mat: Tensor, indices: List[int]) -> Tensor:
    """
    Fill the specified indices in a matrix with 1.

    Args:
        mat (Tensor): The matrix to fill.
        indices (List[int]): List of indices to fill.

    Returns:
        Tensor: Updated matrix with specified indices filled.
    """
    for i in indices:
        for j in indices:
            mat[i, j] = 1
    return mat


def get_iso_matrix() -> Tensor:
    """
    Generate an isomorphism matrix for irreducible spaces.

    Returns:
        Tensor: Isomorphism matrix of shape [9, 9].
    """
    iso_matrix = torch.eye(9, 9)

    # Define isomorphic spaces
    iso_matrix = fill_indices(iso_matrix, [4, 5, 6])  # n-1 dimensional representations
    iso_matrix = fill_indices(iso_matrix, [0, 2])  # W0+ and W1+
    iso_matrix = fill_indices(iso_matrix, [1, 3])  # W0- and W1-

    return iso_matrix
