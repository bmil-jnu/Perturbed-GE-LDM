"""
Optimizer and learning rate scheduler builders.
"""

from typing import List, Optional

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ..models.nn_utils import NoamLR


def build_optimizer(
    model: nn.Module,
    optimizer_type: str = "adam",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    **kwargs
) -> Optimizer:
    """
    Build optimizer for model.
    
    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer ('adam', 'adamw')
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        **kwargs: Additional optimizer arguments
        
    Returns:
        Initialized optimizer
    """
    params = [{"params": model.parameters(), "lr": learning_rate, "weight_decay": weight_decay}]
    
    if optimizer_type.lower() == "adam":
        return Adam(params, **kwargs)
    elif optimizer_type.lower() == "adamw":
        return AdamW(params, **kwargs)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def build_lr_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "noam",
    warmup_epochs: int = 2,
    total_epochs: int = 200,
    steps_per_epoch: int = 100,
    init_lr: float = 1e-4,
    max_lr: float = 1e-3,
    final_lr: float = 1e-4,
    **kwargs
) -> _LRScheduler:
    """
    Build learning rate scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler ('noam', 'cosine', 'polynomial')
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of epochs
        steps_per_epoch: Number of steps per epoch
        init_lr: Initial learning rate
        max_lr: Maximum learning rate
        final_lr: Final learning rate
        **kwargs: Additional scheduler arguments
        
    Returns:
        Initialized learning rate scheduler
    """
    if scheduler_type.lower() == "noam":
        return NoamLR(
            optimizer=optimizer,
            warmup_epochs=[warmup_epochs],
            total_epochs=[total_epochs],
            steps_per_epoch=steps_per_epoch,
            init_lr=[init_lr],
            max_lr=[max_lr],
            final_lr=[final_lr],
        )
    elif scheduler_type.lower() == "cosine":
        try:
            from transformers import get_cosine_schedule_with_warmup
            total_steps = total_epochs * steps_per_epoch
            warmup_steps = warmup_epochs * steps_per_epoch
            return get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )
        except ImportError:
            raise ImportError("transformers package required for cosine scheduler")
    elif scheduler_type.lower() == "polynomial":
        try:
            from transformers import get_polynomial_decay_schedule_with_warmup
            total_steps = total_epochs * steps_per_epoch
            warmup_steps = warmup_epochs * steps_per_epoch
            return get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                lr_end=final_lr,
            )
        except ImportError:
            raise ImportError("transformers package required for polynomial scheduler")
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def build_optimizer_and_scheduler(
    model: nn.Module,
    train_data_size: int,
    batch_size: int,
    epochs: int = 200,
    warmup_epochs: int = 2,
    init_lr: float = 1e-4,
    max_lr: float = 1e-3,
    final_lr: float = 1e-4,
    optimizer_type: str = "adam",
    scheduler_type: str = "noam",
) -> tuple:
    """
    Build both optimizer and scheduler together.
    
    Args:
        model: Model to optimize
        train_data_size: Size of training dataset
        batch_size: Training batch size
        epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        init_lr: Initial learning rate
        max_lr: Maximum learning rate
        final_lr: Final learning rate
        optimizer_type: Type of optimizer
        scheduler_type: Type of scheduler
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    steps_per_epoch = max(train_data_size // batch_size, 1)
    
    optimizer = build_optimizer(
        model=model,
        optimizer_type=optimizer_type,
        learning_rate=init_lr,
    )
    
    scheduler = build_lr_scheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        warmup_epochs=warmup_epochs,
        total_epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        init_lr=init_lr,
        max_lr=max_lr,
        final_lr=final_lr,
    )
    
    return optimizer, scheduler
