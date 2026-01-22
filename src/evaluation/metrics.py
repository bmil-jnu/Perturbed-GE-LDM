"""
Evaluation metrics for gene expression prediction.
"""

from typing import Tuple, Dict
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error


def pearson_mean(predictions: np.ndarray, targets: np.ndarray) -> Tuple[float, float]:
    """
    Compute mean Pearson correlation across samples.
    
    For each sample, computes Pearson correlation between predicted and true
    gene expression vectors, then averages across samples.
    
    Args:
        predictions: Predicted values (N, num_genes)
        targets: Target values (N, num_genes)
        
    Returns:
        Tuple of (mean_correlation, mean_pvalue)
    """
    n_samples = predictions.shape[0]
    sum_corr = 0.0
    sum_pval = 0.0
    valid_count = 0
    
    for i in range(n_samples):
        pred_i = predictions[i]
        targ_i = targets[i]
        
        # Skip samples with NaN or Inf values
        if np.any(~np.isfinite(pred_i)) or np.any(~np.isfinite(targ_i)):
            continue
        
        corr, pval = pearsonr(pred_i, targ_i)
        if np.isfinite(corr):
            sum_corr += corr
            sum_pval += pval
            valid_count += 1
    
    if valid_count == 0:
        return 0.0, 1.0
    
    return sum_corr / valid_count, sum_pval / valid_count


def r2_mean(targets: np.ndarray, predictions: np.ndarray) -> float:
    """
    Compute mean R² score across samples.
    
    For each sample, computes R² between predicted and true gene expression,
    then averages across samples.
    
    Args:
        targets: Target values (N, num_genes)
        predictions: Predicted values (N, num_genes)
        
    Returns:
        Mean R² score
    """
    n_samples = targets.shape[0]
    sum_r2 = 0.0
    valid_count = 0
    
    for i in range(n_samples):
        pred_i = predictions[i]
        targ_i = targets[i]
        
        # Skip samples with NaN or Inf values
        if np.any(~np.isfinite(pred_i)) or np.any(~np.isfinite(targ_i)):
            continue
        
        r2 = r2_score(targ_i, pred_i)
        if np.isfinite(r2):
            sum_r2 += r2
            valid_count += 1
    
    if valid_count == 0:
        return 0.0
    
    return sum_r2 / valid_count


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compute comprehensive metrics for gene expression prediction.
    
    Args:
        predictions: Predicted values (N, num_genes)
        targets: Target values (N, num_genes)
        
    Returns:
        Dictionary of metrics
    """
    # Sample-level metrics (averaged across samples)
    avg_pearson, avg_pval = pearson_mean(predictions, targets)
    avg_r2 = r2_mean(targets, predictions)
    
    # Global metrics (across all values)
    global_mse = mean_squared_error(targets.flatten(), predictions.flatten())
    global_rmse = root_mean_squared_error(targets.flatten(), predictions.flatten())
    
    # Gene-level metrics
    n_genes = predictions.shape[1]
    gene_pearsons = []
    gene_r2s = []
    
    for j in range(n_genes):
        corr, _ = pearsonr(predictions[:, j], targets[:, j])
        gene_pearsons.append(corr)
        gene_r2s.append(r2_score(targets[:, j], predictions[:, j]))
    
    # Mean/Var level R² (captures distribution matching)
    pred_mean = predictions.mean(axis=0)
    pred_var = predictions.var(axis=0)
    target_mean = targets.mean(axis=0)
    target_var = targets.var(axis=0)
    
    r2_mean_level = r2_score(target_mean, pred_mean)
    r2_var_level = r2_score(target_var, pred_var)
    
    return {
        # Sample-level (each sample is a gene expression profile)
        "avg_sample_pearson": avg_pearson,
        "avg_sample_r2": avg_r2,
        
        # Gene-level (each gene across samples)
        "avg_gene_pearson": np.mean(gene_pearsons),
        "avg_gene_r2": np.mean(gene_r2s),
        "median_gene_pearson": np.median(gene_pearsons),
        "median_gene_r2": np.median(gene_r2s),
        
        # Global metrics
        "mse": global_mse,
        "rmse": global_rmse,
        
        # Distribution metrics
        "r2_mean_level": r2_mean_level,
        "r2_var_level": r2_var_level,
    }


def pearson_per_gene(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Compute Pearson correlation for each gene.
    
    Args:
        predictions: Predicted values (N, num_genes)
        targets: Target values (N, num_genes)
        
    Returns:
        Array of correlations for each gene (num_genes,)
    """
    n_genes = predictions.shape[1]
    correlations = np.zeros(n_genes)
    
    for j in range(n_genes):
        corr, _ = pearsonr(predictions[:, j], targets[:, j])
        correlations[j] = corr
    
    return correlations


def r2_per_gene(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """
    Compute R² score for each gene.
    
    Args:
        predictions: Predicted values (N, num_genes)
        targets: Target values (N, num_genes)
        
    Returns:
        Array of R² scores for each gene (num_genes,)
    """
    n_genes = predictions.shape[1]
    scores = np.zeros(n_genes)
    
    for j in range(n_genes):
        scores[j] = r2_score(targets[:, j], predictions[:, j])
    
    return scores
