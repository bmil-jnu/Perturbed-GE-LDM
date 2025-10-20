import logging
from typing import Callable
from sklearn.metrics import root_mean_squared_error

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
import torch.distributed as dist

from tqdm import tqdm
import wandb

from args import TrainArgs
from data.Dataset import DrugDoseAnnDataset

from models.model import MoleculeModel
from GE_VAE.VAE_model import GE_VAE

from datetime import datetime


def train(model: MoleculeModel,
          ge_vae: GE_VAE,
          data_loader: DataLoader,
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: TrainArgs,
          n_iter: int = 0,
          logger: logging.Logger = None,
          ) -> int:
    """
    Trains a model for an epoch.

    :param model: A :class:`~model.MoleculeModel`.
    :param data_loader: A :class:`torch.utils.data.DataLoader`.
    :param loss_func: Loss function.
    :param optimizer: An optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: A :class:`~args.TrainArgs` object containing arguments for training the model.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for recording output.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print

    model.train()
    ge_vae.eval()

    loss_sum = iter_count = rmse_sum = 0
    timesteps = args.timesteps

    debug(f"[DEBUG] Starting training")

    for batch in tqdm(data_loader, total=len(data_loader), leave=False):
        # Prepare batch
        batch: DrugDoseAnnDataset

        t1 = datetime.now()

        basal_ge, smiles, dose, time, cell = batch['features']
        targets = batch['targets']

        batch_size = len(smiles)

        targets = torch.tensor([[0 if x is None else x for x in tb]
                               for tb in targets])  # shape(batch, tasks)
        # Move targets to the model's device
        targets = targets.to(model.device)

        z_0, _, _ = ge_vae.encode(targets)  # shape(batch, latent_dim)

        basal_ge = basal_ge.to(model.device)  # shape(batch, features)

        time = time.to(model.device)  # shape(batch, 1)
        dose = dose.to(model.device)  # shape(batch, 1)

        # Run model
        model.zero_grad()

        t = torch.randint(0, timesteps, (batch_size,), device=model.device)
        z_t, noise = model.module.forward_diffusion(z_0, t)

        # 2. calculate posterior
        posterior_mean, posterior_log_var = model.module.forward_posterior(
            z_0, z_t, t)

        if args.mode == "pred_mu_var":
            pred_mu, pred_log_var = model.module.denoiser(
                z_t, basal_ge, smiles, dose, time, t
            )
        elif args.mode == "pred_mu_v":
            pred_mu, pred_log_var, v, log_beta_t, log_beta_tilde = model.module.denoiser_pred_v(
                z_t, basal_ge, smiles, dose, time, t
            )

        z_t_minus_1 = torch.sqrt(model.module.alpha_bars[t - 1].view(-1, 1)) * z_0 + \
            torch.sqrt(
                1 - model.module.alpha_bars[t - 1].view(-1, 1)) * torch.randn_like(z_0)

        # # Move tensors to correct device
        torch_device = pred_mu.device
        targets = targets.to(torch_device)

        recon_weight = 0.1
        # train_loss = loss_func(pred_mu, pred_log_var, z_t_minus_1)
        if args.loss_function == 'variational_loss':
            train_loss = loss_func(pred_mu, pred_log_var, z_t_minus_1)
        elif args.loss_function == 'hybrid_loss':
            train_loss = loss_func(pred_mu, pred_log_var,
                                   posterior_mean, posterior_log_var)

        wandb.log({"Train/pred_mu": pred_mu.mean().item(),
                   "Train/pred_log_var": pred_log_var.mean().item(),
                   "Train/z_t_minus_1_mu": z_t_minus_1.mean().item(),
                   "Train/z_t_minus_1_std": z_t_minus_1.std().item(),
                   "Train/posterior_mean": posterior_mean.mean().item(),
                   "Train/posterior_log_var": posterior_log_var.mean().item(),
                   "Train/v": v.mean().item() if args.mode == "pred_mu_v" else 0,
                   "Train/log_beta_t": log_beta_t.mean().item() if args.mode == "pred_mu_v" else 0,
                   "Train/log_beta_tilde": log_beta_tilde.mean().item() if args.mode == "pred_mu_v" else 0,
                   })
        loss_sum += train_loss.item()

        iter_count += 1

        train_loss.backward()

        if args.grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        scheduler.step()

        n_iter += len(batch)

        wandb.log({"Train/lr": round(scheduler.get_lr()[0], 8)})

    if dist.get_rank() == 0:  # Only log from rank 0
        wandb.log({"Train/epoch": args.epochs,
                   "Train/loss": loss_sum / iter_count})

    return n_iter
