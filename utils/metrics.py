import numpy as np
import torch
from typing import Dict

def _to_np(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)

def mae(y_true, y_pred) -> float:
    y_true, y_pred = _to_np(y_true), _to_np(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _to_np(y_true), _to_np(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    y_true, y_pred = _to_np(y_true), _to_np(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_pred - y_true) / denom)))

def r2(y_true, y_pred) -> float:
    y_true, y_pred = _to_np(y_true), _to_np(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2) + 1e-8
    return float(1.0 - ss_res / ss_tot)

def relative_error(y_true, y_pred, eps: float = 1e-8) -> float:
    """Alias for MAPE (commonly used for RUL relative error)."""
    return mape(y_true, y_pred, eps)

def evaluate_soh_rul(soh_true, soh_pred, rul_true, rul_pred) -> Dict[str, float]:
    return {
        "SOH_MAE": mae(soh_true, soh_pred),
        "SOH_RMSE": rmse(soh_true, soh_pred),
        "SOH_R2": r2(soh_true, soh_pred),
        "RUL_MAE": mae(rul_true, rul_pred),
        "RUL_RMSE": rmse(rul_true, rul_pred),
        "RUL_RE": relative_error(rul_true, rul_pred)
    }
