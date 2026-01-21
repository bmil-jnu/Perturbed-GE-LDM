"""
Loss functions for training.
"""

from typing import Callable, Tuple

import torch
import torch.nn as nn


def get_loss_func(loss_function: str) -> Callable:
    """
    Get loss function by name.
    
    Args:
        loss_function: Name of loss function
        
    Returns:
        Loss function callable
    """
    loss_functions = {
        "mse": nn.MSELoss(reduction="mean"),
        "variational_loss": variational_loss,
        "hybrid_loss": hybrid_loss,
    }
    
    if loss_function not in loss_functions:
        raise ValueError(
            f'Loss function "{loss_function}" not supported. '
            f'Available: {list(loss_functions.keys())}'
        )
    
    return loss_functions[loss_function]


def variational_loss(
    pred_mu: torch.Tensor,
    log_var: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Variational loss (negative log likelihood under Gaussian).
    
    Args:
        pred_mu: Predicted mean
        log_var: Predicted log variance
        target: Target values
        
    Returns:
        Scalar loss
    """
    precision = torch.exp(-log_var)
    loss = 0.5 * (precision * (target - pred_mu) ** 2 + log_var)
    return loss.mean()


def normal_kl(
    mean1: torch.Tensor,
    logvar1: torch.Tensor,
    mean2: torch.Tensor,
    logvar2: torch.Tensor
) -> torch.Tensor:
    """
    Compute KL divergence between two Gaussians.
    
    KL(N(mean1, var1) || N(mean2, var2))
    
    Args:
        mean1: Mean of first Gaussian
        logvar1: Log variance of first Gaussian
        mean2: Mean of second Gaussian
        logvar2: Log variance of second Gaussian
        
    Returns:
        KL divergence (element-wise)
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )


def hybrid_loss(
    pred_mu: torch.Tensor,
    pred_log_var: torch.Tensor,
    true_mu: torch.Tensor,
    true_log_var: torch.Tensor,
    lambda_kl: float = 0.05
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Hybrid loss combining MSE and variational lower bound.
    
    Args:
        pred_mu: Predicted mean
        pred_log_var: Predicted log variance
        true_mu: True mean
        true_log_var: True log variance
        lambda_kl: Weight for KL term
        
    Returns:
        Tuple of (total_loss, mse_loss, vlb_loss)
    """
    # Loss for Mean: MSE Loss
    loss_mu = torch.nn.functional.mse_loss(pred_mu, true_mu.detach())
    
    # Loss for Variance: Variational Lower Bound (VLB) Loss
    kl_div = normal_kl(
        true_mu.detach(), true_log_var.detach(),  # true posterior
        pred_mu.detach(), pred_log_var            # predicted
    )
    
    loss_vlb = kl_div.mean()
    
    return loss_mu + lambda_kl * loss_vlb, loss_mu, loss_vlb


def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Simple MSE loss."""
    return torch.nn.functional.mse_loss(pred, target)
