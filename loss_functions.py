from typing import Callable

import torch
import torch.nn as nn
import numpy as np

from args import TrainArgs


def get_loss_func(args: TrainArgs) -> Callable:
    """
    Gets the loss function corresponding to a given dataset type.

    :param args: Arguments containing the dataset type ("classification", "regression", or "multiclass").
    :return: A PyTorch loss function.
    """

    # Nested dictionary of the form {dataset_type: {loss_function: loss_function callable}}
    supported_loss_functions = {
        "regression": {
            "mse": nn.MSELoss(reduction="mean"),
            'variational_loss': variational_loss,
            'hybrid_loss': hybrid_loss,
        },
    }

    # Error if no loss function supported
    if args.dataset_type not in supported_loss_functions.keys():
        raise ValueError(f'Dataset type "{args.dataset_type}" not supported.')

    # Return loss function if it is represented in the supported_loss_functions dictionary
    loss_function_choices = supported_loss_functions.get(args.dataset_type, dict())
    loss_function = loss_function_choices.get(args.loss_function)

    if loss_function is not None:
        return loss_function

    else:
        raise ValueError(
            f'Loss function "{args.loss_function}" not supported with dataset type {args.dataset_type}. \
            Available options for that dataset type are {loss_function_choices.keys()}.'
        )


def variational_loss(pred_mu, log_var, target):
    precision = torch.exp(-log_var)
    loss = 0.5 * (precision * (target - pred_mu) ** 2 + log_var)
    return loss.mean()


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for torch.exp().
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


def hybrid_loss(pred_mu, pred_log_var, true_mu, true_log_var, lambda_kl=0.05):
    # Loss for Mean: MSE Loss
    loss_mu = torch.nn.functional.mse_loss(pred_mu, true_mu.detach())
    
    # Loss for Variance: Variational Lower Bound (VLB) Loss
    true_mu = true_mu.detach()
    true_log_var = true_log_var.detach()
    
    kl_div = normal_kl(
        true_mu.detach(), true_log_var.detach(),  # true posterior
        pred_mu.detach(), pred_log_var            # predicted
    )
    
    loss_vlb = kl_div.mean()
    
    print("loss_mu:", loss_mu.item(), "loss_vlb:", loss_vlb.item(), 'lambda_kl:', lambda_kl)

    return loss_mu + lambda_kl * loss_vlb


