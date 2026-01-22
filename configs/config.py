"""
Configuration classes for LDM-LINCS using dataclasses.
Provides type-safe configuration management with YAML support.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional, Any, Dict
import yaml
import json
import os


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    
    # Diffusion settings
    timesteps: int = 1000
    beta_schedule: Literal["cosine", "linear"] = "linear"
    n_steps: int = 50  # Sampling steps
    mode: Literal["pred_mu_var", "pred_mu_v"] = "pred_mu_v"
    
    # Architecture
    latent_dim: int = 256
    hidden_dim: int = 512
    context_dim: int = 576
    ge_dim: int = 978
    
    # Compound embedding
    comp_embed_model: str = "molformer"
    
    # Checkpoint paths
    ldm_path: str = "./checkpoints/ldm/best_ldm.pt"
    vae_path: str = "./checkpoints/vae/best_vae.pt"
    vae_latent_dim: int = 256


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    
    # Basic training
    epochs: int = 200
    batch_size: int = 256
    seed: int = 0
    
    # Learning rate schedule
    init_lr: float = 1e-4
    max_lr: float = 1e-3
    final_lr: float = 1e-4
    warmup_epochs: int = 2
    
    # Loss function
    loss_function: Literal["mse", "variational_loss", "hybrid_loss"] = "variational_loss"
    lambda_kl: float = 0.0001
    
    # Regularization
    grad_clip: Optional[float] = None
    
    # Early stopping
    early_stop_patience: int = 30
    metric: str = "avg_gene_pearson"
    
    # Distributed training
    parallel: bool = False
    
    # Checkpointing
    save_intermediate: bool = False
    save_preds: bool = True
    restart_from_checkpoint: bool = False


@dataclass
class DataConfig:
    """Data loading configuration."""
    
    # Data paths
    data_path: str = "./Lincs_L1000.h5ad"
    cache_dir: str = "./cache"
    
    # Split settings
    split_keys: List[str] = field(default_factory=lambda: ["4foldcv_0"])
    obs_key: str = "cov_drug_name"
    
    # Data options
    data_sample: bool = False
    sample_size: Optional[int] = None
    
    # DataLoader settings
    num_workers: int = 8
    pin_memory: bool = True


@dataclass
class ExperimentConfig:
    """Main experiment configuration combining all sub-configs."""
    
    # Sub-configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Experiment settings
    experiment_name: str = "ldm_lincs"
    save_dir: str = "./checkpoints"
    
    # Mode
    mode: Literal["train", "eval", "predict"] = "train"
    
    # Device settings
    device: str = "cuda"
    cuda: bool = True
    
    # Logging
    quiet: bool = False
    use_wandb: bool = True
    wandb_project: str = "LDM"
    
    # Runtime (set during execution)
    rank: int = 0
    world_size: int = 1
    model_idx: int = 0
    
    def __post_init__(self):
        """Validate and process configuration after initialization."""
        # Ensure save_dir exists
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Adjust batch size for distributed training
        if self.training.parallel and self.world_size > 1:
            self.training.batch_size = self.training.batch_size * self.world_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        # Extract sub-configs
        model_dict = config_dict.pop("model", {})
        training_dict = config_dict.pop("training", {})
        data_dict = config_dict.pop("data", {})
        
        return cls(
            model=ModelConfig(**model_dict),
            training=TrainingConfig(**training_dict),
            data=DataConfig(**data_dict),
            **config_dict
        )
    
    def save(self, path: str) -> None:
        """Save configuration to file (YAML or JSON)."""
        config_dict = self.to_dict()
        
        if path.endswith(".yaml") or path.endswith(".yml"):
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        else:
            with open(path, "w") as f:
                json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load configuration from file (YAML or JSON)."""
        if path.endswith(".yaml") or path.endswith(".yml"):
            with open(path, "r") as f:
                config_dict = yaml.safe_load(f)
        else:
            with open(path, "r") as f:
                config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def update(self, **kwargs) -> "ExperimentConfig":
        """Update config with new values using dot notation."""
        for key, value in kwargs.items():
            if "." in key:
                # Handle nested keys like "model.timesteps"
                parts = key.split(".")
                obj = self
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                if hasattr(self, key):
                    setattr(self, key, value)
        return self


def load_config(path: str) -> ExperimentConfig:
    """Load configuration from file."""
    return ExperimentConfig.load(path)


def save_config(config: ExperimentConfig, path: str) -> None:
    """Save configuration to file."""
    config.save(path)


def get_default_config() -> ExperimentConfig:
    """Get default configuration."""
    return ExperimentConfig()
