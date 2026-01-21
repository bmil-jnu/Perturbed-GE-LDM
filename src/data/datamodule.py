"""
DataModule for LINCS L1000 dataset.
Handles data loading, splitting, and DataLoader creation.
"""

from typing import Optional, Tuple, Dict, Any
import os

import scanpy as sc
import torch
from torch.utils.data import DataLoader, DistributedSampler

from .dataset import DrugDoseAnnDataset, precompute_mol_embeddings
from .utils import train_valid_test


class LINCSDataModule:
    """
    DataModule that handles all data-related operations for LINCS L1000 dataset.
    
    This class encapsulates:
    - Data loading from AnnData files
    - Train/validation/test splitting
    - MolFormer embedding precomputation
    - DataLoader creation (with optional distributed sampling)
    
    Example:
        >>> datamodule = LINCSDataModule(config.data, split_key="4foldcv_0")
        >>> datamodule.setup()
        >>> train_loader = datamodule.train_dataloader(batch_size=256)
    """
    
    def __init__(
        self,
        data_path: str,
        split_key: str,
        obs_key: str = "cov_drug_name",
        cache_dir: str = "./cache",
        data_sample: bool = False,
        num_workers: int = 8,
        pin_memory: bool = True,
    ):
        """
        Initialize DataModule.
        
        Args:
            data_path: Path to AnnData file (.h5ad)
            split_key: Key in adata.obs for train/valid/test split
            obs_key: Key in adata.obs for covariate information
            cache_dir: Directory for caching embeddings
            data_sample: Whether to sample a subset of data
            num_workers: Number of workers for DataLoader
            pin_memory: Whether to pin memory for DataLoader
        """
        self.data_path = data_path
        self.split_key = split_key
        self.obs_key = obs_key
        self.cache_dir = cache_dir
        self.data_sample = data_sample
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Will be set during setup
        self.adata = None
        self._train_adata = None
        self._val_adata = None
        self._test_adata = None
        self._ctrl_len = 0
        
        # MolFormer embeddings
        self._mol_embeddings = None
        self._smiles_to_idx = None
        
        # Datasets
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        
        self._is_setup = False
    
    @property
    def var_names(self):
        """Get variable (gene) names from AnnData."""
        if self.adata is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return self.adata.var_names
    
    @property
    def num_genes(self) -> int:
        """Get number of genes."""
        return len(self.var_names)
    
    @property
    def train_size(self) -> int:
        """Get training data size (excluding controls)."""
        if self._train_adata is None:
            return 0
        return len(self._train_adata) - self._ctrl_len
    
    @property
    def val_size(self) -> int:
        """Get validation data size (excluding controls)."""
        if self._val_adata is None:
            return 0
        return len(self._val_adata) - self._ctrl_len
    
    @property
    def test_size(self) -> int:
        """Get test data size (excluding controls)."""
        if self._test_adata is None:
            return 0
        return len(self._test_adata) - self._ctrl_len
    
    def setup(self, device: torch.device = None) -> None:
        """
        Load data and prepare datasets.
        
        Args:
            device: Device for computing MolFormer embeddings
        """
        if self._is_setup:
            return
        
        # Load AnnData
        print(f"Loading AnnData from {self.data_path}...")
        self.adata = sc.read(self.data_path)
        
        # Split data
        print(f"Splitting data using key: {self.split_key}")
        self._train_adata, self._val_adata, self._test_adata, self._ctrl_len = \
            train_valid_test(self.adata, split_key=self.split_key, sample=self.data_sample)
        
        print(f"Data sizes (excluding controls):")
        print(f"  Train: {self.train_size}")
        print(f"  Validation: {self.val_size}")
        print(f"  Test: {self.test_size}")
        
        # Precompute MolFormer embeddings
        self._precompute_embeddings(device)
        
        # Create datasets
        self._create_datasets()
        
        self._is_setup = True
    
    def _precompute_embeddings(self, device: torch.device = None) -> None:
        """Precompute MolFormer embeddings for all SMILES."""
        print("Precomputing MolFormer embeddings...")
        
        # Collect all SMILES from all datasets
        all_smiles = []
        for adata in [self._train_adata, self._val_adata, self._test_adata]:
            if adata is not None:
                obs = adata.obs
                control_vals = obs["control"].to_numpy()
                drug_mask = control_vals == 0
                smiles_list = obs.loc[drug_mask, "SMILES"].astype(str).tolist()
                all_smiles.extend(smiles_list)
        
        # Compute embeddings (with caching)
        cache_path = os.path.join(self.cache_dir, "mol_embeddings.pt")
        self._mol_embeddings, self._smiles_to_idx = precompute_mol_embeddings(
            all_smiles,
            device=device,
            cache_path=cache_path
        )
    
    def _create_datasets(self) -> None:
        """Create PyTorch datasets from AnnData."""
        print("Creating datasets...")
        
        if self._train_adata is not None:
            self._train_dataset = DrugDoseAnnDataset(
                self._train_adata,
                obs_key=self.obs_key,
                mol_embeddings=self._mol_embeddings,
                smiles_to_idx=self._smiles_to_idx,
            )
        
        if self._val_adata is not None:
            self._val_dataset = DrugDoseAnnDataset(
                self._val_adata,
                obs_key=self.obs_key,
                mol_embeddings=self._mol_embeddings,
                smiles_to_idx=self._smiles_to_idx,
            )
        
        if self._test_adata is not None:
            self._test_dataset = DrugDoseAnnDataset(
                self._test_adata,
                obs_key=self.obs_key,
                mol_embeddings=self._mol_embeddings,
                smiles_to_idx=self._smiles_to_idx,
            )
    
    def train_dataloader(
        self,
        batch_size: int,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ) -> DataLoader:
        """
        Get training DataLoader.
        
        Args:
            batch_size: Batch size
            distributed: Whether to use distributed sampling
            rank: Process rank for distributed training
            world_size: Total number of processes
            seed: Random seed for sampler
            
        Returns:
            Training DataLoader
        """
        if self._train_dataset is None:
            raise RuntimeError("Training dataset not available. Call setup() first.")
        
        if distributed:
            sampler = DistributedSampler(
                dataset=self._train_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
                seed=seed,
            )
            return DataLoader(
                self._train_dataset,
                sampler=sampler,
                batch_size=batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        
        return DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def val_dataloader(
        self,
        batch_size: int,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
    ) -> DataLoader:
        """
        Get validation DataLoader.
        
        Args:
            batch_size: Batch size
            distributed: Whether to use distributed sampling
            rank: Process rank for distributed training
            world_size: Total number of processes
            seed: Random seed for sampler
            
        Returns:
            Validation DataLoader
        """
        if self._val_dataset is None:
            raise RuntimeError("Validation dataset not available. Call setup() first.")
        
        if distributed:
            sampler = DistributedSampler(
                dataset=self._val_dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False,
                seed=seed,
            )
            return DataLoader(
                self._val_dataset,
                sampler=sampler,
                batch_size=batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
            )
        
        return DataLoader(
            self._val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(
        self,
        batch_size: int,
        distributed: bool = False,
        rank: int = 0,
        world_size: int = 1,
    ) -> DataLoader:
        """
        Get test DataLoader.
        
        Args:
            batch_size: Batch size
            distributed: Whether to use distributed sampling
            rank: Process rank for distributed training
            world_size: Total number of processes
            
        Returns:
            Test DataLoader
        """
        if self._test_dataset is None:
            raise RuntimeError("Test dataset not available. Call setup() first.")
        
        return DataLoader(
            self._test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
    
    def get_datasets(self) -> Dict[str, Optional[DrugDoseAnnDataset]]:
        """Get all datasets as a dictionary."""
        return {
            "train": self._train_dataset,
            "valid": self._val_dataset,
            "test": self._test_dataset,
        }
    
    def teardown(self) -> None:
        """Clean up resources."""
        self.adata = None
        self._train_adata = None
        self._val_adata = None
        self._test_adata = None
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None
        self._mol_embeddings = None
        self._smiles_to_idx = None
        self._is_setup = False
