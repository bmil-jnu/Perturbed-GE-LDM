"""
Training callbacks for checkpointing and early stopping.
"""

from typing import Optional
import os

import torch
import torch.nn as nn


class EarlyStopping:
    """
    Early stopping callback to stop training when validation metric stops improving.
    
    Example:
        >>> early_stopping = EarlyStopping(patience=10)
        >>> for epoch in range(100):
        ...     val_score = validate()
        ...     if val_score > best_score:
        ...         early_stopping.reset()
        ...     else:
        ...         early_stopping.step()
        ...         if early_stopping.should_stop:
        ...             break
    """
    
    def __init__(self, patience: int = 30):
        """
        Initialize EarlyStopping.
        
        Args:
            patience: Number of epochs to wait before stopping
        """
        self.patience = patience
        self.counter = 0
        self.should_stop = False
    
    def step(self) -> None:
        """Increment counter (no improvement)."""
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
    
    def reset(self) -> None:
        """Reset counter (improvement detected)."""
        self.counter = 0
        self.should_stop = False
    
    @property
    def count(self) -> int:
        """Get current counter value."""
        return self.counter


class ModelCheckpoint:
    """
    Callback for saving model checkpoints.
    
    Example:
        >>> checkpoint = ModelCheckpoint(save_dir="./checkpoints")
        >>> checkpoint.save(model, optimizer, scheduler, epoch, score)
    """
    
    def __init__(
        self,
        save_dir: str,
        filename: str = "model.pt",
        save_best_only: bool = True,
    ):
        """
        Initialize ModelCheckpoint.
        
        Args:
            save_dir: Directory to save checkpoints
            filename: Checkpoint filename
            save_best_only: Whether to only save best model
        """
        self.save_dir = save_dir
        self.filename = filename
        self.save_best_only = save_best_only
        self.best_score = float("-inf")
        
        os.makedirs(save_dir, exist_ok=True)
    
    @property
    def path(self) -> str:
        """Get full checkpoint path."""
        return os.path.join(self.save_dir, self.filename)
    
    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[object] = None,
        epoch: int = 0,
        best_score: float = 0.0,
        early_stop_count: int = 0,
        scaler: Optional[dict] = None,
        config: Optional[dict] = None,
    ) -> None:
        """
        Save checkpoint.
        
        Args:
            model: Model to save
            optimizer: Optional optimizer
            scheduler: Optional scheduler
            epoch: Current epoch
            best_score: Best validation score
            early_stop_count: Early stopping counter
            scaler: Optional data scaler
            config: Optional configuration
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
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "random": random.getstate(),
            },
        }
        
        torch.save(state, self.path)
        print(f"Checkpoint saved: {self.path}")
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[object] = None,
        device: torch.device = None,
    ) -> dict:
        """
        Load checkpoint.
        
        Args:
            model: Model to load weights into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            device: Device to load to
            
        Returns:
            Checkpoint state dictionary
        """
        import random
        import numpy as np
        
        state = torch.load(
            self.path,
            map_location=device or "cpu",
            weights_only=False
        )
        
        # Load model
        loaded_state_dict = state["state_dict"]
        
        # Remove 'module.' prefix if present
        for key in list(loaded_state_dict.keys()):
            if key.startswith("module."):
                loaded_state_dict[key[7:]] = loaded_state_dict.pop(key)
        
        model.load_state_dict(loaded_state_dict)
        
        # Load optimizer
        if optimizer and state.get("optimizer"):
            optimizer.load_state_dict(state["optimizer"])
        
        # Load scheduler
        if scheduler and state.get("scheduler"):
            scheduler.load_state_dict(state["scheduler"])
        
        # Load random state
        if state.get("random_state"):
            rs = state["random_state"]
            torch.set_rng_state(rs["torch"])
            if rs["cuda"] and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rs["cuda"])
            np.random.set_state(rs["numpy"])
            random.setstate(rs["random"])
        
        print(f"Checkpoint loaded: {self.path}")
        
        return state
