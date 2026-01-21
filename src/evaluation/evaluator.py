"""
Evaluator class for model evaluation.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from .metrics import pearson_mean, r2_mean, compute_metrics
from ..training.loss import variational_loss


class Evaluator:
    """
    Evaluator class for computing metrics on validation/test sets.
    
    Handles:
    - Diffusion sampling for prediction
    - Metric computation (Pearson, R2, MSE, RMSE)
    - Multi-GPU aggregation
    - Results logging
    
    Example:
        >>> evaluator = Evaluator(model, vae, config)
        >>> metrics = evaluator.evaluate(test_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        vae: nn.Module,
        device: torch.device = None,
        n_steps: int = 50,
        mode: str = "pred_mu_v",
        distributed: bool = False,
        scaler: Optional[Dict] = None,
    ):
        """
        Initialize Evaluator.
        
        Args:
            model: Diffusion model
            vae: VAE model
            device: Device for computation
            n_steps: Number of sampling steps
            mode: Prediction mode
            distributed: Whether running distributed
            scaler: Optional data scaler for inverse transform
        """
        self.model = model
        self.vae = vae
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_steps = n_steps
        self.mode = mode
        self.distributed = distributed
        self.scaler = scaler
    
    def _get_model_module(self) -> nn.Module:
        """Get underlying model (handles DDP wrapper)."""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False,
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            dataloader: DataLoader for evaluation data
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary of metrics (and predictions if requested)
        """
        self.model.eval()
        self.vae.eval()
        
        model_module = self._get_model_module()
        
        # Accumulators
        all_predictions = []
        all_targets = []
        
        val_loss_sum = 0.0
        gene_pearson_sum = 0.0
        gene_r2_sum = 0.0
        gene_mse_sum = 0.0
        gene_rmse_sum = 0.0
        r2_mean_sum = 0.0
        r2_var_sum = 0.0
        count = 0
        
        for batch in tqdm(dataloader, leave=False, desc="Evaluating"):
            batch_metrics, preds, targets = self._evaluate_batch(batch, model_module)
            
            val_loss_sum += batch_metrics["val_loss"]
            gene_pearson_sum += batch_metrics["gene_pearson"]
            gene_r2_sum += batch_metrics["gene_r2"]
            gene_mse_sum += batch_metrics["gene_mse"]
            gene_rmse_sum += batch_metrics["gene_rmse"]
            r2_mean_sum += batch_metrics["r2_mean"]
            r2_var_sum += batch_metrics["r2_var"]
            count += 1
            
            if return_predictions:
                all_predictions.append(preds)
                all_targets.append(targets)
        
        # Average metrics
        metrics = {
            "val_loss": val_loss_sum / max(count, 1),
            "avg_gene_pearson": gene_pearson_sum / max(count, 1),
            "avg_gene_r2": gene_r2_sum / max(count, 1),
            "avg_gene_mse": gene_mse_sum / max(count, 1),
            "avg_gene_rmse": gene_rmse_sum / max(count, 1),
            "r2_mean": r2_mean_sum / max(count, 1),
            "r2_var": r2_var_sum / max(count, 1),
        }
        
        # Aggregate across GPUs if distributed
        if self.distributed and dist.is_initialized():
            metrics = self._aggregate_metrics(metrics)
        
        if return_predictions:
            predictions = np.concatenate(all_predictions, axis=0)
            targets = np.concatenate(all_targets, axis=0)
            return metrics, predictions, targets
        
        return metrics
    
    def _evaluate_batch(
        self,
        batch: Dict,
        model_module: nn.Module,
    ) -> tuple:
        """
        Evaluate single batch.
        
        Args:
            batch: Batch data
            model_module: Model module
            
        Returns:
            Tuple of (metrics_dict, predictions, targets)
        """
        from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
        
        # Unpack batch
        basal_ge, smiles, dose, time, cell, mol_emb = batch["features"]
        targets = batch["targets"]
        
        batch_size = len(smiles)
        
        # Move to device
        targets = targets.to(self.device)
        basal_ge = basal_ge.to(self.device)
        time = time.to(self.device)
        dose = dose.to(self.device)
        
        if mol_emb is not None:
            mol_emb = mol_emb.to(self.device)
        
        # Encode targets
        z_0, _, _ = self.vae.encode(targets)
        
        timesteps = model_module.n_timesteps
        t = torch.randint(0, timesteps, (batch_size,), device=self.device)
        
        # Forward diffusion and get posterior
        z_t, noise = model_module.forward_diffusion(z_0, t)
        z_t_minus_1 = (
            torch.sqrt(model_module.alpha_bars[t - 1].view(-1, 1)) * z_0 +
            torch.sqrt(1 - model_module.alpha_bars[t - 1].view(-1, 1)) * torch.randn_like(z_0)
        )
        
        # Predict
        if self.mode == "pred_mu_var":
            pred_mu, log_var = model_module.denoiser(
                z_t, basal_ge, smiles, dose, time, t, mol_emb_cached=mol_emb
            )
        else:
            pred_mu, log_var, _, _, _ = model_module.denoiser_pred_v(
                z_t, basal_ge, smiles, dose, time, t, mol_emb_cached=mol_emb
            )
        
        # Sample from model
        if self.mode == "pred_mu_var":
            samples, mu, sigma = model_module.sample(
                cell, basal_ge, smiles, dose, time, batch_size,
                n_steps=self.n_steps, mol_emb_cached=mol_emb
            )
        else:
            samples, mu, sigma, _, _, _ = model_module.sample(
                cell, basal_ge, smiles, dose, time, batch_size,
                n_steps=self.n_steps, mol_emb_cached=mol_emb
            )
        
        # Decode samples
        samples = self.vae.decode(samples)
        
        # Convert to numpy
        samples_np = samples.data.cpu().numpy()
        targets_np = targets.cpu().numpy()
        
        # Inverse scale if scaler provided
        if self.scaler is not None:
            if isinstance(self.scaler, dict):
                std = self.scaler["stds"]
                mean = self.scaler["means"]
            else:
                std = self.scaler.stds
                mean = self.scaler.means
            samples_np = samples_np * std + mean
            targets_np = targets_np * std + mean
        
        samples_np = samples_np.astype(float)
        
        # Compute metrics
        gene_pearson, _ = pearson_mean(samples_np, targets_np)
        gene_r2 = r2_mean(targets_np, samples_np)
        gene_mse = mean_squared_error(targets_np, samples_np)
        gene_rmse = root_mean_squared_error(targets_np, samples_np)
        
        # Variational loss
        val_loss = variational_loss(pred_mu, log_var, z_t_minus_1).item()
        
        # Mean/Var level R2
        yp_m = samples_np.mean(0)
        yp_v = samples_np.var(0)
        yt_m = targets_np.mean(axis=0)
        yt_v = targets_np.var(axis=0)
        r2_m = r2_score(yt_m, yp_m)
        r2_v = r2_score(yt_v, yp_v)
        
        metrics = {
            "val_loss": val_loss,
            "gene_pearson": gene_pearson,
            "gene_r2": gene_r2,
            "gene_mse": gene_mse,
            "gene_rmse": gene_rmse,
            "r2_mean": r2_m,
            "r2_var": r2_v,
        }
        
        return metrics, samples_np, targets_np
    
    def _aggregate_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Aggregate metrics across distributed processes."""
        aggregated = {}
        for key, value in metrics.items():
            tensor = torch.tensor([value], device=self.device)
            gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, tensor)
            aggregated[key] = np.mean([t.item() for t in gathered])
        return aggregated
