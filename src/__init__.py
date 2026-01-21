"""
LDM-LINCS: Latent Diffusion Model for LINCS L1000 gene expression prediction.

This package provides a refactored, modular implementation of the LDM for
predicting drug-induced gene expression changes.

Modules:
    - configs: Configuration management
    - data: Data loading and preprocessing
    - models: Neural network architectures
    - training: Training loop and utilities
    - evaluation: Metrics and evaluation
    - utils: General utilities
"""

__version__ = "1.0.0"
__author__ = "LDM-LINCS Team"

# Convenience imports
from configs import ExperimentConfig, load_config
from .data import LINCSDataModule
from .models import ModelFactory, LatentDiffusionModel, GE_VAE
from .training import Trainer
from .evaluation import Evaluator, Predictor

__all__ = [
    "ExperimentConfig",
    "load_config",
    "LINCSDataModule",
    "ModelFactory",
    "LatentDiffusionModel",
    "GE_VAE",
    "Trainer",
    "Evaluator",
    "Predictor",
]
