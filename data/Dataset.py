from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data.utils import StandardScaler
from scipy import sparse
import scanpy as sc

class DrugDoseAnnDataset(Dataset):
    """
    Dataset for drug perturbation experiments with conditions.
    dense_adata: sc.AnnData,
        AnnData object containing gene expression data and metadata.
    obs_key: str, default "cov_drug"
        Key in adata.obs for covariate information.
    device: Optional[torch.device], default None
        Device to store tensors on.
    copy_X: bool, default False
        Whether to copy the data matrix X.
    """

    def __init__(
        self,
        dense_adata: sc.AnnData,
        obs_key: str = "cov_drug",
        device: Optional[torch.device] = None,
        copy_X: bool = False,
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
        
        dose_vals = obs["dose"].to_numpy()
        drug_mask = dose_vals != 0.0
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

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "features": (
                self.controls[index],
                self.smiles_list[index],
                self.dose_tensor[index],
                self.time_tensor[index],
                self.cell_list[index],
            ),
            "targets": self.data[index],
            "cov_drug": self.obs_list[index],
        }


    # ---------------------------
    # Targets normalization utils
    # ---------------------------
    def normalize_targets(self) -> StandardScaler:
        """
        Fit StandardScaler on targets(self.data) and transform in-place.
        반환: fitted StandardScaler
        """
        # torch → list-of-arrays 로 변환 (기존 StandardScaler와 호환)
        targets = [row.cpu().numpy() for row in self.data]
        scaler = StandardScaler().fit(targets)
        scaled = scaler.transform(targets)
        # 다시 텐서에 반영
        self.data = torch.from_numpy(np.asarray(scaled, dtype=np.float32))
        return scaler


    def set_targets(self, targets) -> None:
        """
        targets: 길이 __len__() 와 동일한 2D 배열/리스트. self.data를 통째로 교체.
        """
        targets = np.asarray(targets, dtype=np.float32)
        if targets.shape != tuple(self.data.shape):
            raise ValueError(
                f"targets shape mismatch: got {targets.shape}, expected {tuple(self.data.shape)}"
            )
        self.data = torch.from_numpy(targets)
