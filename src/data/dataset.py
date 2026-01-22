"""
Dataset classes for drug perturbation experiments.
"""

from typing import Optional, Dict, Any
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy import sparse
import scanpy as sc

from .transforms import StandardScaler


def precompute_mol_embeddings(
    smiles_list: list,
    device: torch.device = None,
    cache_path: str = None,
) -> tuple:
    """
    Precompute MolFormer embeddings for a list of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings
        device: Device to compute embeddings on
        cache_path: Path to cache embeddings (optional)
    
    Returns:
        Tuple of (embeddings tensor, smiles_to_idx dict)
    """
    from transformers import AutoModel, AutoTokenizer
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if cached embeddings exist
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached MolFormer embeddings from {cache_path}")
        cached = torch.load(cache_path, weights_only=True)
        return cached['embeddings'], cached['smiles_to_idx']
    
    print("Precomputing MolFormer embeddings...")
    
    # Get unique SMILES to avoid redundant computation
    unique_smiles = list(set(smiles_list))
    smiles_to_idx = {s: i for i, s in enumerate(unique_smiles)}
    
    # Load MolFormer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    molformer = AutoModel.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct",
        trust_remote_code=True,
        deterministic_eval=True
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        "ibm/MoLFormer-XL-both-10pct", 
        trust_remote_code=True
    )
    
    # Compute embeddings in batches
    batch_size = 256
    all_embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(unique_smiles), batch_size), desc="Computing MolFormer embeddings"):
            batch_smiles = unique_smiles[i:i + batch_size]
            inputs = tokenizer(batch_smiles, padding=True, return_tensors="pt").to(device)
            embeddings = molformer(**inputs).pooler_output  # (B, 768)
            all_embeddings.append(embeddings.cpu())
    
    unique_embeddings = torch.cat(all_embeddings, dim=0)  # (num_unique, 768)
    
    # Clean up MolFormer from GPU memory
    del molformer
    torch.cuda.empty_cache()
    
    # Cache the embeddings
    if cache_path:
        print(f"Caching MolFormer embeddings to {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({
            'embeddings': unique_embeddings,
            'smiles_to_idx': smiles_to_idx
        }, cache_path)
    
    print(f"Computed {len(unique_smiles)} unique MolFormer embeddings")
    return unique_embeddings, smiles_to_idx


class DrugDoseAnnDataset(Dataset):
    """
    Dataset for drug perturbation experiments with conditions.
    
    Args:
        dense_adata: AnnData object containing gene expression data and metadata
        obs_key: Key in adata.obs for covariate information
        device: Device to store tensors on
        copy_X: Whether to copy the data matrix X
        mol_embeddings: Precomputed MolFormer embeddings tensor
        smiles_to_idx: Dictionary mapping SMILES to embedding indices
    """

    def __init__(
        self,
        dense_adata: sc.AnnData,
        obs_key: str = "cov_drug",
        device: Optional[torch.device] = None,
        copy_X: bool = False,
        mol_embeddings: torch.Tensor = None,
        smiles_to_idx: dict = None,
    ):
        super().__init__()
        self.obs_key = obs_key
        self.device = device
        obs = dense_adata.obs
        
        X = dense_adata.X
        if sparse.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        if copy_X:
            X = X.copy()
        np.nan_to_num(X, copy=False, nan=0.0, posinf=None, neginf=None)
        
        dose_vals = obs["dose"].to_numpy()
        control_vals = obs["control"].to_numpy()
        drug_mask = control_vals == 0  # if control == 0, it's drug-treated sample
        self.drug_idx = np.flatnonzero(drug_mask)
        
        ctrl_keys = obs.loc[drug_mask, "paired_control_index"].to_numpy()
        
        obs_names = obs.index.to_numpy()
        name_to_pos = {}
        for i, name in enumerate(obs_names):
            if name not in name_to_pos:
                name_to_pos[name] = i
        
        ctrl_pos = np.array([name_to_pos[k] for k in ctrl_keys], dtype=np.int32)
        
        self.data = torch.from_numpy(X[self.drug_idx, :])   # perturbed GE
        self.controls = torch.from_numpy(X[ctrl_pos, :])    # basal GE
        
        drug_obs = obs.iloc[self.drug_idx]
        
        self.cell_list = drug_obs["cell_id"].astype(str).tolist()
        self.smiles_list = drug_obs["SMILES"].astype(str).tolist()
        self.obs_list = drug_obs[obs_key].tolist()
        
        self.dose_tensor = torch.from_numpy(
            dose_vals[self.drug_idx].astype(np.float32)
        ).view(-1, 1)  # (N, 1)
        
        self.time_tensor = torch.from_numpy(
            drug_obs["pert_time"].astype(np.float32).to_numpy()
        ).view(-1, 1)  # (N, 1)
        
        # Store precomputed MolFormer embeddings
        self.mol_embeddings = mol_embeddings  # (num_unique, 768)
        self.smiles_to_idx = smiles_to_idx    # dict: smiles -> index
        
        # Create index mapping for each sample
        if mol_embeddings is not None and smiles_to_idx is not None:
            self.mol_emb_indices = torch.tensor(
                [smiles_to_idx[s] for s in self.smiles_list],
                dtype=torch.long
            )
        else:
            self.mol_emb_indices = None

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Get precomputed MolFormer embedding if available
        if self.mol_embeddings is not None and self.mol_emb_indices is not None:
            mol_emb = self.mol_embeddings[self.mol_emb_indices[index]]
        else:
            mol_emb = None
            
        return {
            "features": (
                self.controls[index],
                self.smiles_list[index],
                self.dose_tensor[index],
                self.time_tensor[index],
                self.cell_list[index],
                mol_emb,  # Precomputed MolFormer embedding (768,) or None
            ),
            "targets": self.data[index],
            "cov_drug": self.obs_list[index],
        }

    def normalize_targets(self) -> StandardScaler:
        """
        Fit StandardScaler on targets (self.data) and transform in-place.
        
        Returns:
            Fitted StandardScaler
        """
        targets = [row.cpu().numpy() for row in self.data]
        scaler = StandardScaler().fit(targets)
        scaled = scaler.transform(targets)
        self.data = torch.from_numpy(np.asarray(scaled, dtype=np.float32))
        return scaler

    def set_targets(self, targets) -> None:
        """
        Set targets directly.
        
        Args:
            targets: 2D array of shape matching self.data
        """
        targets = np.asarray(targets, dtype=np.float32)
        if targets.shape != tuple(self.data.shape):
            raise ValueError(
                f"targets shape mismatch: got {targets.shape}, expected {tuple(self.data.shape)}"
            )
        self.data = torch.from_numpy(targets)
