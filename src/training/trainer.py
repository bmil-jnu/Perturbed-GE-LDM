"""
Trainer class for managing the training loop.
"""

from typing import Dict, Optional, Any, Callable, List
import os

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .callbacks import EarlyStopping, ModelCheckpoint
from ..models.factory import ModelFactory


class Trainer:
    """
    Trainer class that manages the training loop.
    
    Handles:
    - Training and validation loops
    - Loss computation
    - Optimizer and scheduler updates
    - Checkpointing and early stopping
    - Distributed training
    - Logging (wandb)
    
    Example:
        >>> trainer = Trainer(
        ...     model=model,
        ...     vae=vae,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler,
        ...     loss_fn=loss_fn,
        ...     config=config,
        ... )
        >>> trainer.fit(train_loader, val_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        vae: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        loss_fn: Callable,
        config: Any,
        device: torch.device = None,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        use_wandb: bool = True,
        evaluator: Any = None,
    ):
        """
        Initialize Trainer.
        
        Args:
            model: The diffusion model to train
            vae: The VAE for encoding/decoding
            optimizer: Optimizer for training
            scheduler: Learning rate scheduler
            loss_fn: Loss function
            config: Experiment configuration
            device: Device for training
            distributed: Whether using distributed training
            rank: Process rank
            world_size: Total number of processes
            use_wandb: Whether to log to wandb
            evaluator: Optional evaluator for validation
        """
        self.model = model
        self.vae = vae
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.evaluator = evaluator
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_score = float("-inf")
        self.best_epoch = 0
        
        # Callbacks
        self.early_stopping = None
        self.checkpoint = None
        
        # Check if main process
        self.is_main_process = not distributed or rank == 0
        
    def _setup_callbacks(self, save_dir: str, patience: int = 30) -> None:
        """Set up training callbacks."""
        self.early_stopping = EarlyStopping(patience=patience)
        self.checkpoint = ModelCheckpoint(
            save_dir=save_dir,
            filename="model.pt"
        )
    
    def _log(self, metrics: Dict[str, float], prefix: str = "") -> None:
        """Log metrics to wandb."""
        if self.use_wandb and self.is_main_process:
            log_dict = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
            wandb.log(log_dict)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None,
        save_dir: str = "./checkpoints",
        early_stop_patience: int = 30,
        metric: str = "avg_gene_pearson",
    ) -> Dict[str, float]:
        """
        Run the training loop.
        
        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            num_epochs: Number of epochs (uses config if not provided)
            save_dir: Directory to save checkpoints
            early_stop_patience: Patience for early stopping
            metric: Metric to monitor for early stopping
            
        Returns:
            Dictionary of final metrics
        """
        num_epochs = num_epochs or getattr(self.config.training, 'epochs', 200)
        
        # Setup callbacks
        self._setup_callbacks(save_dir, early_stop_patience)
        
        # Training loop
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Set epoch for distributed sampler
            if self.distributed and hasattr(train_loader, 'sampler'):
                if hasattr(train_loader.sampler, 'set_epoch'):
                    train_loader.sampler.set_epoch(epoch)
                    val_loader.sampler.set_epoch(epoch)
            
            # Training
            train_metrics = self._train_epoch(train_loader, epoch)
            self._log(train_metrics, prefix="Train")
            
            # Validation
            val_metrics = self._validate(val_loader)
            self._log(val_metrics, prefix="Valid")
            
            # Check for improvement and save checkpoint
            current_score = val_metrics.get(metric, val_metrics.get("avg_gene_pearson", 0))
            
            if self.is_main_process:
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_epoch = epoch
                    
                    # Save checkpoint
                    self.checkpoint.save(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        best_score=self.best_score,
                    )
                    
                    self.early_stopping.reset()
                else:
                    self.early_stopping.step()
                    
                    if self.early_stopping.should_stop:
                        print(f"Early stopping at epoch {epoch}")
                        break
            
            # Sync early stopping across processes
            if self.distributed:
                should_stop = torch.tensor(
                    1 if self.early_stopping.should_stop else 0,
                    device=self.device
                )
                dist.broadcast(should_stop, src=0)
                if should_stop.item() == 1:
                    break
        
        print(f"Best validation {metric} = {self.best_score:.6f} at epoch {self.best_epoch}")
        
        return {"best_score": self.best_score, "best_epoch": self.best_epoch}
    
    def _train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            loader: Training DataLoader
            epoch: Current epoch number
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        self.vae.eval()
        
        total_loss = 0.0
        iter_count = 0
        timesteps = getattr(self.config.model, 'timesteps', 1000)
        mode = getattr(self.config.model, 'mode', 'pred_mu_v')
        loss_function_name = getattr(self.config.training, 'loss_function', 'variational_loss')
        
        progress_bar = tqdm(loader, total=len(loader), leave=False, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            loss = self._train_step(batch, timesteps, mode, loss_function_name)
            total_loss += loss
            iter_count += 1
            self.global_step += 1
            
            progress_bar.set_postfix({"loss": f"{loss:.4f}"})
        
        avg_loss = total_loss / max(iter_count, 1)
        
        return {
            "loss": avg_loss,
            "epoch": epoch,
            "lr": self.scheduler.get_lr()[0] if hasattr(self.scheduler, 'get_lr') else 0,
        }
    
    def _train_step(
        self,
        batch: Dict,
        timesteps: int,
        mode: str,
        loss_function_name: str
    ) -> float:
        """
        Execute single training step.
        
        Args:
            batch: Batch of data
            timesteps: Number of diffusion timesteps
            mode: Prediction mode
            loss_function_name: Name of loss function
            
        Returns:
            Loss value
        """
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
        
        # Encode targets to latent space
        with torch.no_grad():
            z_0, _, _ = self.vae.encode(targets)
        
        # Forward pass
        self.optimizer.zero_grad()
        
        # Sample random timesteps
        t = torch.randint(0, timesteps, (batch_size,), device=self.device)
        
        # Get model (handle DDP)
        model_module = self.model.module if isinstance(self.model, DDP) else self.model
        
        # Forward diffusion
        z_t, noise = model_module.forward_diffusion(z_0, t)
        
        # Calculate posterior
        posterior_mean, posterior_log_var = model_module.forward_posterior(z_0, z_t, t)
        
        # Predict
        if mode == "pred_mu_var":
            pred_mu, pred_log_var = model_module.denoiser(
                z_t, basal_ge, smiles, dose, time, t, mol_emb_cached=mol_emb
            )
        elif mode == "pred_mu_v":
            pred_mu, pred_log_var, v, log_beta_t, log_beta_tilde = model_module.denoiser_pred_v(
                z_t, basal_ge, smiles, dose, time, t, mol_emb_cached=mol_emb
            )
        
        # Compute loss
        if loss_function_name == "variational_loss":
            train_loss = self.loss_fn(pred_mu, pred_log_var, posterior_mean)
        elif loss_function_name == "hybrid_loss":
            lambda_kl = getattr(self.config.training, 'lambda_kl', 0.0001)
            train_loss, loss_mu, loss_vlb = self.loss_fn(
                pred_mu, pred_log_var, posterior_mean, posterior_log_var, lambda_kl
            )
        elif loss_function_name == "mse":
            train_loss = self.loss_fn(pred_mu, posterior_mean)
        else:
            train_loss = self.loss_fn(pred_mu, pred_log_var, posterior_mean)
        
        # Backward pass
        train_loss.backward()
        
        # Gradient clipping
        grad_clip = getattr(self.config.training, 'grad_clip', None)
        if grad_clip:
            nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
        
        self.optimizer.step()
        self.scheduler.step()
        
        return train_loss.item()
    
    def _validate(self, loader: DataLoader) -> Dict[str, float]:
        """
        Run validation.
        
        Args:
            loader: Validation DataLoader
            
        Returns:
            Dictionary of validation metrics
        """
        if self.evaluator is not None:
            return self.evaluator.evaluate(loader)
        
        # Simple validation if no evaluator provided
        self.model.eval()
        
        total_loss = 0.0
        count = 0
        timesteps = getattr(self.config.model, 'timesteps', 1000)
        mode = getattr(self.config.model, 'mode', 'pred_mu_v')
        
        with torch.no_grad():
            for batch in tqdm(loader, leave=False, desc="Validation"):
                basal_ge, smiles, dose, time, cell, mol_emb = batch["features"]
                targets = batch["targets"]
                
                batch_size = len(smiles)
                
                targets = targets.to(self.device)
                basal_ge = basal_ge.to(self.device)
                time = time.to(self.device)
                dose = dose.to(self.device)
                
                if mol_emb is not None:
                    mol_emb = mol_emb.to(self.device)
                
                z_0, _, _ = self.vae.encode(targets)
                
                t = torch.randint(0, timesteps, (batch_size,), device=self.device)
                
                model_module = self.model.module if isinstance(self.model, DDP) else self.model
                z_t, _ = model_module.forward_diffusion(z_0, t)
                posterior_mean, posterior_log_var = model_module.forward_posterior(z_0, z_t, t)
                
                if mode == "pred_mu_var":
                    pred_mu, pred_log_var = model_module.denoiser(
                        z_t, basal_ge, smiles, dose, time, t, mol_emb_cached=mol_emb
                    )
                else:
                    pred_mu, pred_log_var, _, _, _ = model_module.denoiser_pred_v(
                        z_t, basal_ge, smiles, dose, time, t, mol_emb_cached=mol_emb
                    )
                
                loss = self.loss_fn(pred_mu, pred_log_var, posterior_mean)
                total_loss += loss.item()
                count += 1
        
        avg_loss = total_loss / max(count, 1)
        
        return {"loss": avg_loss, "avg_gene_pearson": -avg_loss}  # Placeholder
    
    def save_checkpoint(self, path: str) -> None:
        """Save current model state."""
        ModelFactory.save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            best_score=self.best_score,
        )
    
    def load_checkpoint(self, path: str) -> None:
        """Load model state from checkpoint."""
        model, scaler, state = ModelFactory.load_checkpoint(
            path,
            device=self.device,
            model=self.model.module if isinstance(self.model, DDP) else self.model
        )
        
        if isinstance(self.model, DDP):
            self.model.module.load_state_dict(model.state_dict())
        else:
            self.model.load_state_dict(model.state_dict())
        
        if state.get("optimizer"):
            self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler"):
            self.scheduler.load_state_dict(state["scheduler"])
        
        self.current_epoch = state.get("epoch", 0)
        self.best_score = state.get("best_score", float("-inf"))
