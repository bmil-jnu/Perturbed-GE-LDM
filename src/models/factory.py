"""
Model factory for creating and loading models.
"""

from typing import Tuple, Optional, Dict, Any
import os

import torch
import torch.nn as nn

from .diffusion import LatentDiffusionModel
from .vae import GE_VAE


class ModelFactory:
    """
    Factory class for creating and loading models.
    
    This class provides a unified interface for:
    - Creating new models from configuration
    - Loading pretrained models from checkpoints
    - Managing model initialization
    """
    
    @staticmethod
    def create_diffusion_model(
        device: torch.device,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        mode: str = "pred_mu_v",
        latent_dim: int = 256,
        hidden_dim: int = 512,
        context_dim: int = 576,
        ge_dim: int = 978,
        model_idx: int = 0,
        **kwargs
    ) -> LatentDiffusionModel:
        """
        Create a new diffusion model (LatentDiffusionModel).
        
        Args:
            device: Device to place model on
            timesteps: Number of diffusion steps
            beta_schedule: Beta scheduling method ('cosine' or 'linear')
            mode: Prediction mode ('pred_mu_var' or 'pred_mu_v')
            latent_dim: Latent dimension size
            hidden_dim: Hidden layer dimension
            context_dim: Context embedding dimension
            ge_dim: Gene expression dimension
            model_idx: Model index for identification
            **kwargs: Additional arguments
            
        Returns:
            Initialized LatentDiffusionModel
        """
        # Create args-like object for backward compatibility
        class ModelArgs:
            pass
        
        args = ModelArgs()
        args.device = device
        args.timesteps = timesteps
        args.beta_schedule = beta_schedule
        args.mode = mode
        args.model_idx = model_idx
        
        model = LatentDiffusionModel(
            args,
            context_dim=context_dim,
            ge_dim=ge_dim,
            hidden_dim=hidden_dim,
            timesteps=timesteps
        )
        
        return model.to(device)
    
    @staticmethod
    def create_vae(
        device: torch.device,
        input_dim: int = 978,
        latent_dim: int = 256,
        checkpoint_path: Optional[str] = None,
        freeze: bool = True,
    ) -> GE_VAE:
        """
        Create or load a GE_VAE model.
        
        Args:
            device: Device to place model on
            input_dim: Input dimension (gene expression size)
            latent_dim: Latent space dimension
            checkpoint_path: Path to pretrained checkpoint
            freeze: Whether to freeze VAE parameters
            
        Returns:
            Initialized or loaded GE_VAE
        """
        vae = GE_VAE(input_dim=input_dim, latent_dim=latent_dim)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading GE_VAE from {checkpoint_path}")
            state = torch.load(checkpoint_path, weights_only=True)
            vae.load_state_dict(state["vae"])
        
        vae = vae.to(device)
        vae.eval()
        
        if freeze:
            for param in vae.parameters():
                param.requires_grad = False
        
        return vae
    
    @staticmethod
    def create_models(
        device: torch.device,
        model_config: Dict[str, Any],
    ) -> Tuple[LatentDiffusionModel, GE_VAE]:
        """
        Create both diffusion model and VAE.
        
        Args:
            device: Device to place models on
            model_config: Model configuration dictionary
            
        Returns:
            Tuple of (LatentDiffusionModel, GE_VAE)
        """
        # Create diffusion model
        diffusion_model = ModelFactory.create_diffusion_model(
            device=device,
            timesteps=model_config.get("timesteps", 1000),
            beta_schedule=model_config.get("beta_schedule", "linear"),
            mode=model_config.get("mode", "pred_mu_v"),
            latent_dim=model_config.get("latent_dim", 256),
            hidden_dim=model_config.get("hidden_dim", 512),
            context_dim=model_config.get("context_dim", 576),
            ge_dim=model_config.get("ge_dim", 978),
        )
        
        # Create VAE
        vae = ModelFactory.create_vae(
            device=device,
            input_dim=model_config.get("ge_dim", 978),
            latent_dim=model_config.get("vae_latent_dim", 256),
            checkpoint_path=model_config.get("vae_path"),
            freeze=True,
        )
        
        return diffusion_model, vae
    
    @staticmethod
    def load_checkpoint(
        checkpoint_path: str,
        device: torch.device,
        model: Optional[LatentDiffusionModel] = None,
    ) -> Tuple[LatentDiffusionModel, Optional[Dict], Dict[str, Any]]:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to place model on
            model: Optional existing model to load weights into
            
        Returns:
            Tuple of (model, scaler_dict, checkpoint_state)
        """
        print(f"Loading checkpoint from {checkpoint_path}")
        
        state = torch.load(
            checkpoint_path,
            map_location=lambda storage, loc: storage,
            weights_only=False
        )
        
        loaded_state_dict = state["state_dict"]
        data_scaler = state.get("data_scaler")
        
        # Remove 'module.' prefix if present (from DDP)
        for key in list(loaded_state_dict.keys()):
            if key.startswith("module."):
                new_key = key[7:]
                loaded_state_dict[new_key] = loaded_state_dict.pop(key)
        
        if model is None:
            # Need to create model from saved args or config
            args = state.get("args")
            config = state.get("config")
            
            if args is not None:
                # Legacy: use args
                class ModelArgs:
                    pass
                
                model_args = ModelArgs()
                for k, v in vars(args).items():
                    setattr(model_args, k, v)
                model_args.device = device
                
                model = LatentDiffusionModel(model_args)
            elif config is not None:
                # New: use config dict
                model_config = config.get("model", config)
                
                class ModelArgs:
                    pass
                
                model_args = ModelArgs()
                model_args.timesteps = model_config.get("timesteps", 1000)
                model_args.beta_schedule = model_config.get("beta_schedule", "linear")
                model_args.mode = model_config.get("mode", "pred_mu_v")
                model_args.latent_dim = model_config.get("latent_dim", 256)
                model_args.hidden_dim = model_config.get("hidden_dim", 1024)
                model_args.ge_dim = model_config.get("ge_dim", 978)
                model_args.device = device
                
                model = LatentDiffusionModel(model_args)
            else:
                raise ValueError("Cannot create model: no args or config in checkpoint")
        
        # Load weights
        model_state_dict = model.state_dict()
        
        # Filter out incompatible keys
        for key in list(loaded_state_dict.keys()):
            if key not in model_state_dict:
                print(f"Warning: Key '{key}' not found in model, skipping")
                loaded_state_dict.pop(key)
            elif loaded_state_dict[key].shape != model_state_dict[key].shape:
                print(f"Warning: Shape mismatch for '{key}', skipping")
                loaded_state_dict.pop(key)
        
        model_state_dict.update(loaded_state_dict)
        model.load_state_dict(model_state_dict)
        model = model.to(device)
        
        return model, data_scaler, state
    
    @staticmethod
    def save_checkpoint(
        path: str,
        model: nn.Module,
        scaler: Optional[Dict] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        best_score: float = 0.0,
        early_stop_count: int = 0,
        config: Optional[Dict] = None,
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
            model: Model to save
            scaler: Optional data scaler
            optimizer: Optional optimizer
            scheduler: Optional scheduler
            epoch: Current epoch
            best_score: Best validation score
            early_stop_count: Early stopping counter
            config: Optional configuration dictionary
        """
        import random
        import numpy as np
        
        state = {
            "state_dict": model.state_dict(),
            "data_scaler": scaler,
            "optimizer": optimizer.state_dict() if optimizer else None,
            "scheduler": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "best_score": best_score,
            "early_stop_count": early_stop_count,
            "config": config,
            "random_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all(),
                "numpy": np.random.get_state(),
                "random": random.getstate(),
            },
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)
        print(f"Checkpoint saved to {path}")
