"""
Seed utilities for reproducibility.
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """
    Set random seeds for reproducibility.
    
    Sets seeds for:
    - Python random
    - NumPy
    - PyTorch (CPU and CUDA)
    
    Also configures PyTorch for deterministic behavior.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_random_state() -> dict:
    """
    Get current random state for all generators.
    
    Returns:
        Dictionary with random states
    """
    state = {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    
    return state


def set_random_state(state: dict) -> None:
    """
    Restore random state for all generators.
    
    Args:
        state: Dictionary with random states from get_random_state()
    """
    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
