"""
evaluate.py
Evaluation metrics for continuous species-conditioned MIC regression.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error

def compute_all_metrics(y_true, y_pred):
    """
    Computes regression metrics for log10-transformed MIC:
    - MAE
    - RMSE
    - Fold Error (2^MAE)
    - Pearson r
    - Spearman rho
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    fold_error = 2.0 ** mae  # Multiplicative deviation
    
    r_val, _ = pearsonr(y_true, y_pred)
    rho_val, _ = spearmanr(y_true, y_pred)
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "Fold Error": fold_error,
        "Pearson (r)": r_val,
        "Spearman (rho)": rho_val
    }
