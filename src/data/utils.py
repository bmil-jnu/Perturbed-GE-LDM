"""
Data utility functions.
"""

from random import shuffle
from scipy import sparse
from anndata import AnnData
import numpy as np
from typing import Tuple, Optional


def shuffle_adata(adata: AnnData) -> AnnData:
    """
    Shuffle rows of AnnData object.
    
    Args:
        adata: AnnData object to shuffle
        
    Returns:
        Shuffled AnnData object
    """
    if sparse.issparse(adata.X):
        adata.X = adata.X.A

    ind_list = [i for i in range(adata.shape[0])]
    shuffle(ind_list)
    new_adata = adata[ind_list, :]
    return new_adata


def train_valid_test(
    adata: AnnData,
    split_key: str = "cov_drug_dose_name_split",
    sample: bool = False,
    train_sample_size: int = 24000,
    val_sample_size: int = 3000,
    test_sample_size: int = 3000,
) -> Tuple[Optional[AnnData], Optional[AnnData], Optional[AnnData], int]:
    """
    Split AnnData into train, valid, test sets based on split key.
    
    Args:
        adata: AnnData object to split
        split_key: Column in adata.obs containing split labels
        sample: Whether to sample a subset of the data
        train_sample_size: Size of training sample if sample=True
        val_sample_size: Size of validation sample if sample=True
        test_sample_size: Size of test sample if sample=True
        
    Returns:
        Tuple of (train_adata, valid_adata, test_adata, control_count)
    """
    train_index = adata.obs[adata.obs[split_key] == "train"].index.tolist()
    valid_index = adata.obs[adata.obs[split_key] == "valid"].index.tolist()
    test_index = adata.obs[adata.obs[split_key] == "test"].index.tolist()
    
    if sample:
        if len(train_index) > train_sample_size:
            train_index = np.random.choice(train_index, size=train_sample_size, replace=False).tolist()
        if len(valid_index) > val_sample_size:
            valid_index = np.random.choice(valid_index, size=val_sample_size, replace=False).tolist()
        if len(test_index) > test_sample_size:
            test_index = np.random.choice(test_index, size=test_sample_size, replace=False).tolist()
    
    control_index = adata.obs[adata.obs["control"] == 1].index.tolist()

    train_adata = None
    valid_adata = None
    test_adata = None

    if len(train_index) > 0:
        train_index = train_index + control_index
        train_adata = adata[train_index, :]
    
    if len(valid_index) > 0:
        valid_index = valid_index + control_index
        valid_adata = adata[valid_index, :]
    
    if len(test_index) > 0:
        test_index = test_index + control_index
        test_adata = adata[test_index, :]

    return train_adata, valid_adata, test_adata, len(control_index)
