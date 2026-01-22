"""
Predictor class for inference.
"""

from typing import Dict, List, Optional, Tuple, Any
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm


class Predictor:
    """
    Predictor class for running inference with trained models.
    
    Handles:
    - Loading trained models
    - Running inference on new data
    - Saving predictions
    
    Example:
        >>> predictor = Predictor(model, vae, config)
        >>> predictions = predictor.predict(test_loader)
        >>> predictor.save_predictions(predictions, "predictions.csv")
    """
    
    def __init__(
        self,
        model: nn.Module,
        vae: nn.Module,
        device: torch.device = None,
        n_steps: int = 50,
        mode: str = "pred_mu_v",
        scaler: Optional[Dict] = None,
        task_names: Optional[List[str]] = None,
    ):
        """
        Initialize Predictor.
        
        Args:
            model: Trained diffusion model
            vae: VAE model
            device: Device for computation
            n_steps: Number of sampling steps
            mode: Prediction mode
            scaler: Optional data scaler
            task_names: Gene/task names for output
        """
        self.model = model
        self.vae = vae
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_steps = n_steps
        self.mode = mode
        self.scaler = scaler
        self.task_names = task_names
        
        # Set models to eval mode
        self.model.eval()
        self.vae.eval()
    
    def _get_model_module(self) -> nn.Module:
        """Get underlying model (handles DDP wrapper)."""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model
    
    @torch.no_grad()
    def predict(
        self,
        dataloader: DataLoader,
        return_metadata: bool = False,
    ) -> np.ndarray:
        """
        Run inference on dataset.
        
        Args:
            dataloader: DataLoader with input data
            return_metadata: Whether to return metadata (SMILES, cell, etc.)
            
        Returns:
            Predictions array (and metadata if requested)
        """
        model_module = self._get_model_module()
        
        all_predictions = []
        all_metadata = []
        
        for batch in tqdm(dataloader, desc="Predicting"):
            preds, metadata = self._predict_batch(batch, model_module)
            all_predictions.append(preds)
            
            if return_metadata:
                all_metadata.append(metadata)
        
        predictions = np.concatenate(all_predictions, axis=0)
        
        if return_metadata:
            # Combine metadata
            combined_metadata = {
                key: sum([m[key] for m in all_metadata], [])
                for key in all_metadata[0].keys()
            }
            return predictions, combined_metadata
        
        return predictions
    
    def _predict_batch(
        self,
        batch: Dict,
        model_module: nn.Module,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Predict on single batch.
        
        Args:
            batch: Batch data
            model_module: Model module
            
        Returns:
            Tuple of (predictions, metadata)
        """
        # Unpack batch
        basal_ge, smiles, dose, time, cell, mol_emb = batch["features"]
        
        batch_size = len(smiles)
        
        # Move to device
        basal_ge = basal_ge.to(self.device)
        time = time.to(self.device)
        dose = dose.to(self.device)
        
        if mol_emb is not None:
            mol_emb = mol_emb.to(self.device)
        
        # Sample from model
        if self.mode == "pred_mu_var":
            samples, mu, sigma = model_module.sample(
                cell, basal_ge, smiles, dose, time, batch_size,
                n_steps=self.n_steps, mol_emb_cached=mol_emb
            )
        else:
            samples, mu, sigma, _, _, _ = model_module.sample(
                cell, basal_ge, smiles, dose, time, batch_size,
                n_steps=self.n_steps, mol_emb_cached=mol_emb
            )
        
        # Decode samples
        samples = self.vae.decode(samples)
        
        # Convert to numpy
        predictions = samples.data.cpu().numpy()
        
        # Inverse scale if scaler provided
        if self.scaler is not None:
            if isinstance(self.scaler, dict):
                std = self.scaler["stds"]
                mean = self.scaler["means"]
            else:
                std = self.scaler.stds
                mean = self.scaler.means
            predictions = predictions * std + mean
        
        # Collect metadata
        metadata = {
            "smiles": list(smiles) if not isinstance(smiles, list) else smiles,
            "cell": list(cell) if not isinstance(cell, list) else cell,
            "dose": dose.cpu().numpy().flatten().tolist(),
            "time": time.cpu().numpy().flatten().tolist(),
        }
        
        if "cov_drug" in batch:
            metadata["cov_drug"] = (
                list(batch["cov_drug"]) if not isinstance(batch["cov_drug"], list)
                else batch["cov_drug"]
            )
        
        return predictions, metadata
    
    def save_predictions(
        self,
        predictions: np.ndarray,
        save_path: str,
        metadata: Optional[Dict] = None,
        task_names: Optional[List[str]] = None,
    ) -> None:
        """
        Save predictions to CSV file.
        
        Args:
            predictions: Prediction array (N, num_genes)
            save_path: Path to save CSV
            metadata: Optional metadata dictionary
            task_names: Optional gene/task names
        """
        task_names = task_names or self.task_names
        
        # Create directory if needed
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        
        # Create predictions DataFrame
        if task_names is not None:
            pred_df = pd.DataFrame(predictions, columns=task_names)
        else:
            pred_df = pd.DataFrame(
                predictions,
                columns=[f"gene_{i}" for i in range(predictions.shape[1])]
            )
        
        # Add metadata if provided
        if metadata is not None:
            meta_df = pd.DataFrame(metadata)
            df = pd.concat([meta_df, pred_df], axis=1)
        else:
            df = pred_df
        
        # Save to CSV
        df.to_csv(save_path, index=False)
        print(f"Predictions saved to {save_path}")
    
    def save_comparison(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        save_dir: str,
        metadata: Optional[Dict] = None,
        task_names: Optional[List[str]] = None,
    ) -> None:
        """
        Save predictions and ground truth for comparison.
        
        Args:
            predictions: Prediction array
            targets: Ground truth array
            save_dir: Directory to save files
            metadata: Optional metadata
            task_names: Optional gene names
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save predictions
        self.save_predictions(
            predictions,
            os.path.join(save_dir, "predictions.csv"),
            metadata,
            task_names
        )
        
        # Save ground truth
        task_names = task_names or self.task_names
        if task_names is not None:
            truth_df = pd.DataFrame(targets, columns=task_names)
        else:
            truth_df = pd.DataFrame(
                targets,
                columns=[f"gene_{i}" for i in range(targets.shape[1])]
            )
        
        truth_df.to_csv(os.path.join(save_dir, "ground_truth.csv"), index=False)
        print(f"Ground truth saved to {os.path.join(save_dir, 'ground_truth.csv')}")
