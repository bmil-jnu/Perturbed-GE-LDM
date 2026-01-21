"""
Data transformation utilities.
"""

from typing import Any, List, Optional
import numpy as np


class StandardScaler:
    """
    A StandardScaler normalizes the features of a dataset.
    
    When fit on a dataset, learns mean and standard deviation across the 0th axis.
    When transforming, subtracts means and divides by standard deviations.
    """

    def __init__(
        self,
        means: np.ndarray = None,
        stds: np.ndarray = None,
        replace_nan_token: Any = None
    ):
        """
        Initialize scaler.
        
        Args:
            means: Optional precomputed means
            stds: Optional precomputed standard deviations
            replace_nan_token: Token to replace NaN values with
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X: List[List[Optional[float]]]) -> 'StandardScaler':
        """
        Learn means and standard deviations from data.
        
        Args:
            X: 2D array-like of floats (or None)
            
        Returns:
            Fitted scaler (self)
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
        Transform data by subtracting means and dividing by standard deviations.
        
        Args:
            X: 2D array-like of floats (or None)
            
        Returns:
            Transformed array with NaNs replaced
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan),
            self.replace_nan_token,
            transformed_with_nan
        )
        return transformed_with_none

    def inverse_transform(self, X: List[List[Optional[float]]]) -> np.ndarray:
        """
        Inverse transform by multiplying by stds and adding means.
        
        Args:
            X: 2D array-like of floats
            
        Returns:
            Inverse transformed array
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan),
            self.replace_nan_token,
            transformed_with_nan
        )
        return transformed_with_none

    def to_dict(self) -> dict:
        """Convert scaler to dictionary for serialization."""
        return {
            "means": self.means,
            "stds": self.stds,
            "replace_nan_token": self.replace_nan_token,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'StandardScaler':
        """Create scaler from dictionary."""
        return cls(
            means=d.get("means"),
            stds=d.get("stds"),
            replace_nan_token=d.get("replace_nan_token"),
        )
