"""
Neural network utilities.
"""

from typing import List, Union
import numpy as np

import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.
    
    The learning rate increases linearly from init_lr to max_lr over warmup_steps,
    then decreases exponentially from max_lr to final_lr.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: List[Union[float, int]],
        total_epochs: List[int],
        steps_per_epoch: int,
        init_lr: List[float],
        max_lr: List[float],
        final_lr: List[float],
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            warmup_epochs: Number of warmup epochs per param group
            total_epochs: Total number of epochs per param group
            steps_per_epoch: Number of steps per epoch
            init_lr: Initial learning rates
            max_lr: Maximum learning rates
            final_lr: Final learning rates
        """
        if not (
            len(optimizer.param_groups)
            == len(warmup_epochs)
            == len(total_epochs)
            == len(init_lr)
            == len(max_lr)
            == len(final_lr)
        ):
            raise ValueError(
                "Number of param groups must match the number of epochs and learning rates!"
            )

        self.num_lrs = len(optimizer.param_groups)
        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs)
        self.total_epochs = np.array(total_epochs)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr)
        self.max_lr = np.array(max_lr)
        self.final_lr = np.array(final_lr)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (
            1 / (self.total_steps - self.warmup_steps)
        )

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        return list(self.lr)

    def step(self, current_step: int = None):
        """Update learning rate."""
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                # Linear warmup
                self.lr[i] = self.init_lr[i] + self.linear_increment[i] * self.current_step
            elif self.current_step <= self.total_steps[i]:
                # Exponential decay
                self.lr[i] = self.max_lr[i] * (
                    self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i])
                )
            else:
                # After total steps, keep final lr
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]


def param_count(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def param_count_all(model: nn.Module) -> int:
    """Count all parameters in a model (trainable and non-trainable)."""
    return sum(p.numel() for p in model.parameters())
