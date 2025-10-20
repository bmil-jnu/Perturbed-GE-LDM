from random import shuffle
from scipy import sparse
from anndata import AnnData

import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

from typing import Any, List, Optional
import numpy as np


def shuffle_adata(adata):
    """
    Shuffles the `adata`.
    """
    if sparse.issparse(adata.X):
        adata.X = adata.X.A

    ind_list = [i for i in range(adata.shape[0])]
    shuffle(ind_list)
    new_adata = adata[ind_list, :]
    return new_adata


def train_valid_test(adata: AnnData, split_key="cov_drug_dose_name_split", sample=False):
    """
    Get train, valid, test adata based on the split key
    """
    shuffled = shuffle_adata(adata)
    train_index = adata.obs[adata.obs[split_key] == "train"].index.tolist()
    valid_index = adata.obs[adata.obs[split_key] == "valid"].index.tolist()
    test_index = adata.obs[adata.obs[split_key] == "test"].index.tolist()
    
    if sample:
        train_index = np.random.choice(train_index, size=24000, replace=False).tolist()
        valid_index = np.random.choice(valid_index, size=3000, replace=False).tolist()
        test_index = np.random.choice(test_index, size=3000, replace=False).tolist()
    
    control_index = adata.obs[adata.obs["dose"].astype(float) == 0.0].index.tolist()

    if len(train_index) > 0:
        train_index = train_index + control_index
        train_adata = shuffled[train_index, :]
    else:
        train_adata = None
    if len(valid_index) > 0:
        valid_index = valid_index + control_index
        valid_adata = shuffled[valid_index, :]
    else:
        valid_adata = None
    if len(test_index) > 0:
        test_index = test_index + control_index
        test_adata = shuffled[test_index, :]
    else:
        test_adata = None

    return train_adata, valid_adata, test_adata


def Drug_dose_encoder(drug_SMILES_list: list, dose_list: list, num_Bits=1024, comb_num=1):
    """
    Encode SMILES of drug to rFCFP fingerprint
    """
    drug_len = len(drug_SMILES_list)
    fcfp4_array = np.zeros((drug_len, num_Bits))

    for i, smiles in enumerate(drug_SMILES_list):
        smi = smiles
        mol = Chem.MolFromSmiles(smi)
        fcfp4 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, useFeatures=True, nBits=num_Bits).ToBitString()
        fcfp4_list = np.array(list(fcfp4), dtype=np.float32)
        fcfp4_list = fcfp4_list * np.log10(dose_list[i]+1)
        fcfp4_array[i] = fcfp4_list

    return fcfp4_array 


class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.

    When it is fit on a dataset, the :class:`StandardScaler` learns the mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the means and divides by the standard deviations.
    """

    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token: Any = None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X: List[List[Optional[float]]]) -> 'StandardScaler':
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.

        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)

        return self

    def transform(self, X: List[List[Optional[float]]]) -> np.ndarray:
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X: List[List[Optional[float]]]) -> np.ndarray:
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none
