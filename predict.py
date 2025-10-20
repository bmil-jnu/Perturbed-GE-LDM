from typing import List

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from tqdm import tqdm
import numpy as np

from data.Dataset import DrugDoseAnnDataset
from data.utils import StandardScaler
from models.model import MoleculeModel

from utils import pearson_mean, r2_mean

from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from loss_functions import variational_loss, hybrid_loss
import wandb

from GE_VAE.VAE_model import GE_VAE

def all_gather_scalar(value: float, device: torch.device):
    tensor = torch.tensor([value], device=device)
    world_size = dist.get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return [t.item() for t in gathered]


def predict(
    model: MoleculeModel,
    ge_vae: GE_VAE,
    data_loader: DataLoader,
    disable_progress_bar: bool = False,
    scaler: StandardScaler = None,
    args=None,
    test=False,
    debug=None,
    dist_enabled=True,
):
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~model.MoleculeModel`.
    :param data_loader: A :class:`torch.utils.data.DataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~data.utils.StandardScaler` object fit on the training targets.
    :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks. If returning uncertainty parameters as well,
        it is a tuple of lists of lists, of a length depending on how many uncertainty parameters are appropriate for the loss function.
    """
    model.eval()
    ge_vae.eval()  # Ensure the VAE is in evaluation mode
    
    x0_preds = []
    cov_drug_list = []
    
    # Reset metrics
    r2_score_mean = []
    r2_score_var = []
    r2_sum_mean = 0
    r2_sum_var = 0
    val_loss_sum = 0
    gene_pearson_sum = 0
    gene_r2_sum = 0
    gene_mse_sum = 0
    gene_rmse_sum = 0
    
    count = 0

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: DrugDoseAnnDataset
        
        basal_ge, smiles, dose, time, cell = batch['features']
        targets = batch['targets']
        cov_drug = batch['cov_drug']
        cov_drug_list.extend(cov_drug)
        
        
        batch_size = len(smiles)
        
        targets = torch.tensor([[0 if x is None else x for x in tb] for tb in targets]) # shape(batch, tasks)
        targets = targets.to(model.device)  # Move targets to the model's device
        
        with torch.no_grad():
            z_0, _, _ = ge_vae.encode(targets)  # shape(batch, latent_dim)    
                
        basal_ge = basal_ge.to(model.device) # shape(batch, features)

        
        time = time.to(model.device) # shape(batch, 1)
        dose = dose.to(model.device) # shape(batch, 1)
        
        timesteps = args.timesteps
        t = torch.randint(0, timesteps, (batch_size,), device=targets.device)


        if (args.parallel) and (not test):
            z_t_minus_1 = torch.sqrt(model.module.alpha_bars[t - 1].view(-1, 1)) * z_0 + torch.sqrt(1 - model.module.alpha_bars[t - 1].view(-1, 1)) * torch.randn_like(z_0)
            noisy_x, noise_true = model.module.forward_diffusion(z_0, t)
            
            if args.mode == "pred_mu_var":
                pred_mu, log_var = model.module.denoiser(
                    noisy_x, basal_ge, smiles, dose, time, t
                )
            elif args.mode == "pred_mu_v":
                pred_mu, log_var, v, log_beta_t, log_beta_t_tilde = model.module.denoiser_pred_v(
                    noisy_x, basal_ge, smiles, dose, time, t
                )        
        else:
            z_t_minus_1 = torch.sqrt(model.alpha_bars[t - 1].view(-1, 1)) * z_0 + torch.sqrt(1 - model.alpha_bars[t - 1].view(-1, 1)) * torch.randn_like(z_0)
            noisy_x, noise_true = model.forward_diffusion(z_0, t)
            
            if args.mode == "pred_mu_var":
                pred_mu, log_var = model.denoiser(
                    noisy_x, basal_ge, smiles, dose, time, t
                )
            elif args.mode == "pred_mu_v":
                pred_mu, log_var, v, log_beta_t, log_beta_tilde = model.denoiser_pred_v(
                    noisy_x, basal_ge, smiles, dose, time, t
                )

        
        # Make predictions
        save = False
        with torch.no_grad():
            if (args.parallel) and (not test):
                if args.mode == "pred_mu_var":
                    samples, mu, sigma = model.module.sample(cell, basal_ge, smiles, dose, time, batch_size, save=save)
                elif args.mode == "pred_mu_v":
                    samples, mu, sigma, v, log_beta_t, log_beta_tilde = model.module.sample(cell, basal_ge, smiles, dose, time, batch_size, save=save)
            else:
                if args.mode == "pred_mu_var":
                    samples, mu, sigma = model.sample(cell, basal_ge, smiles, dose, time, batch_size, save=save, count=count)
                elif args.mode == "pred_mu_v":
                    samples, mu, sigma, v, log_beta_t, log_beta_tilde = model.sample(cell, basal_ge, smiles, dose, time, batch_size, save=save, count=count)
                count += 1
        
        samples = ge_vae.decode(samples)  # Decode the latent space samples
        
        samples = samples.data.cpu().numpy()
        targets = targets.cpu().numpy()
        
        # Inverse scale if regression
        if scaler is not None:
            if type(scaler) is dict:
                std = scaler['stds']
                mean = scaler['means']
            else:
                std = scaler.stds
                mean = scaler.means
            samples = samples * std + mean
            targets = targets * std + mean
            
        x0_preds.append(samples)

        samples = samples.astype(float)
        
        gene_pearson = pearson_mean(samples, targets)[0]
        gene_r2 = r2_mean(targets, samples)
        gene_mse = mean_squared_error(targets, samples)
        gene_rmse = root_mean_squared_error(targets, samples)
        gene_pearson_sum += gene_pearson
        gene_r2_sum += gene_r2
        gene_mse_sum += gene_mse
        gene_rmse_sum += gene_rmse
        
        # variational loss
        val_loss = variational_loss(pred_mu, log_var, z_t_minus_1)
        val_loss_sum += val_loss.item()
        
        # Mean/Var-level R2
        yp_m = samples.mean(0)
        yp_v = samples.var(0)
        yt_m = targets.mean(axis=0)
        yt_v = targets.var(axis=0)
        r2_sum_mean += r2_score(yt_m, yp_m)
        r2_sum_var += r2_score(yt_v, yp_v)
        
    # Average metrics
    avg_val_loss = val_loss_sum / len(data_loader)
    avg_gene_pearson = gene_pearson_sum / len(data_loader)
    avg_gene_r2 = gene_r2_sum / len(data_loader)
    avg_gene_mse = gene_mse_sum / len(data_loader)
    avg_gene_rmse = gene_rmse_sum / len(data_loader)
    r2_score_mean.append(r2_sum_mean / len(data_loader))
    r2_score_var.append(r2_sum_var / len(data_loader))
    

    rank = 0
    # 2. aggregate across multiple GPUs
    if not test and args.parallel and dist.is_initialized():
        # avg_noise_pearson_list = all_gather_scalar(avg_noise_pearson, model.device)
        # avg_noise_r2_list = all_gather_scalar(avg_noise_r2, model.device)
        # avg_noise_mse_list = all_gather_scalar(avg_noise_mse, model.device)
        avg_val_loss_list = all_gather_scalar(avg_val_loss, model.device)
        avg_gene_pearson_list = all_gather_scalar(avg_gene_pearson, model.device)
        avg_gene_r2_list = all_gather_scalar(avg_gene_r2, model.device)
        avg_gene_mse_list = all_gather_scalar(avg_gene_mse, model.device)
        r2_score_mean_list = all_gather_scalar(r2_score_mean[0], model.device)
        r2_score_var_list = all_gather_scalar(r2_score_var[0], model.device)
        

        if dist.get_rank() == 0:
            # 3. calculate mean across GPUs on rank 0
            results = {
                # 'avg_noise_pearson': np.mean(avg_noise_pearson_list),
                # 'avg_noise_r2': np.mean(avg_noise_r2_list),
                # 'avg_noise_mse': np.mean(avg_noise_mse_list),
                'avg_val_loss': np.mean(avg_val_loss_list),
                'avg_gene_pearson': np.mean(avg_gene_pearson_list),
                'avg_gene_r2': np.mean(avg_gene_r2_list),
                'avg_gene_mse': np.mean(avg_gene_mse_list),
                'r2_score_mean': np.mean(r2_score_mean_list),
                'r2_score_var': np.mean(r2_score_var_list),
                # 'cov_drug_list': cov_drug_list
            }

            debug("Validation Metrics (gathered):")
            for k, v in results.items():
                debug(f"{k}: {v:.4f}")
                wandb.log({f"validation/{k}": v})

        else:
            results = None
    else:
        # Non-DDP fallback
        results = {
            'avg_val_loss': avg_val_loss,
            'avg_gene_pearson': avg_gene_pearson,
            'avg_gene_r2': avg_gene_r2,
            'avg_gene_mse': avg_gene_mse,
            'r2_score_mean': r2_score_mean[0],
            'r2_score_var': r2_score_var[0],
        }
        
        if not test:
            debug("\nValidation Metrics (non-DDP):")
            for k, v in results.items():
                debug(f"{k}: {v:.4f}")
                wandb.log({f"validation/{k}": v})
                rank=dist.get_rank()
        else:
            debug("\nTest Metrics (non-DDP):")
            for k, v in results.items():
                debug(f"{k}: {v:.4f}")
                wandb.log({f"test/{k}": v})

    if save:
        import pandas as pd
        info_df = pd.read_csv('./fold_0/0/info.csv')
        init_z = torch.load(f'./fold_0/{count}/init_z.pt', weights_only=True)
        step_49 = torch.load(f'./fold_0/{count}/step_49.pt', weights_only=True)
        step_39 = torch.load(f'./fold_0/{count}/step_39.pt', weights_only=True)
        step_29 = torch.load(f'./fold_0/{count}/step_29.pt', weights_only=True)
        step_19 = torch.load(f'./fold_0/{count}/step_19.pt', weights_only=True)
        step_9 = torch.load(f'./fold_0/{count}/step_9.pt', weights_only=True)
        step_0 = torch.load(f'./fold_0/{count}/step_0.pt', weights_only=True)
        
        for i in range(1, count-1):
            info_df_i = pd.read_csv(f'./fold_0/{i}/info.csv')
            info_df = pd.concat([info_df, info_df_i], ignore_index=True)
            
            init_z_i = torch.load(f'./fold_0/{i}/init_z.pt', weights_only=True)
            step_49_i = torch.load(f'./fold_0/{i}/step_49.pt', weights_only=True)
            step_39_i = torch.load(f'./fold_0/{i}/step_39.pt', weights_only=True)
            step_29_i = torch.load(f'./fold_0/{i}/step_29.pt', weights_only=True)
            step_19_i = torch.load(f'./fold_0/{i}/step_19.pt', weights_only=True)
            step_9_i = torch.load(f'./fold_0/{i}/step_9.pt', weights_only=True)
            step_0_i = torch.load(f'./fold_0/{i}/step_0.pt', weights_only=True)
            
            init_z = torch.cat([init_z, init_z_i], dim=0)
            step_49 = torch.cat([step_49, step_49_i], dim=0)
            step_39 = torch.cat([step_39, step_39_i], dim=0)
            step_29 = torch.cat([step_29, step_29_i], dim=0)
            step_19 = torch.cat([step_19, step_19_i], dim=0)
            step_9 = torch.cat([step_9, step_9_i], dim=0)
            step_0 = torch.cat([step_0, step_0_i], dim=0)
            
        info_df.reset_index(drop=True, inplace=True)
        info_df.to_csv(f'./fold_0/info_all.csv', index=False)
        import datetime
        mmddhhmm = datetime.datetime.now().strftime("%m%d%H%M")
        torch.save(init_z, f'./fold_0/init_z_{mmddhhmm}.pt')
        torch.save(step_49, f'./fold_0/step_49_{mmddhhmm}.pt')
        torch.save(step_39, f'./fold_0/step_39_{mmddhhmm}.pt')
        torch.save(step_29, f'./fold_0/step_29_{mmddhhmm}.pt')
        torch.save(step_19, f'./fold_0/step_19_{mmddhhmm}.pt')
        torch.save(step_9, f'./fold_0/step_9_{mmddhhmm}.pt')
        torch.save(step_0, f'./fold_0/step_0_{mmddhhmm}.pt')
        

    return results, x0_preds if not dist_enabled or rank == 0 else None