import numpy as np
import pandas as pd
from typing import Tuple, Optional

def interpolate_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Linear interpolation + back/forward fill for missing sensor values.
    """
    return df.interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")

def zscore_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit per-feature z-score on [N, T, F] or [T, F].
    Returns (mean, std).
    """
    if x.ndim == 2:
        mu = x.mean(axis=0)
        sigma = x.std(axis=0) + 1e-8
    else:
        mu = x.reshape(-1, x.shape[-1]).mean(axis=0)
        sigma = x.reshape(-1, x.shape[-1]).std(axis=0) + 1e-8
    return mu, sigma

def zscore_transform(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Apply z-score normalization using provided mean/std."""
    return (x - mu) / sigma

def zscore_fit_transform(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit and transform in one step."""
    mu, sigma = zscore_fit(x)
    return zscore_transform(x, mu, sigma), mu, sigma

def rolling_resample(df: pd.DataFrame, rule: Optional[str] = None, agg: str = "mean") -> pd.DataFrame:
    """
    Resample a time-indexed DataFrame by rule (e.g., '1S', '100ms').
    If rule is None, return df unchanged.
    """
    if rule is None:
        return df
    if agg == "mean":
        return df.resample(rule).mean().dropna()
    if agg == "median":
        return df.resample(rule).median().dropna()
    raise ValueError(f"Unsupported agg: {agg}")

def clip_outliers_iqr(x: np.ndarray, q1: float = 0.25, q3: float = 0.75, k: float = 1.5) -> np.ndarray:
    """
    IQR-based clipping per feature. Robustly limits extreme values.
    """
    x2 = x.copy()
    q1v = np.quantile(x2, q1, axis=0)
    q3v = np.quantile(x2, q3, axis=0)
    iqr = q3v - q1v
    lo = q1v - k * iqr
    hi = q3v + k * iqr
    return np.clip(x2, lo, hi)

def detrend_linear(x: np.ndarray) -> np.ndarray:
    """
    Remove linear trend along time for each feature in [T, F].
    """
    t = np.arange(x.shape[0], dtype=np.float32)
    t = (t - t.mean()) / (t.std() + 1e-8)
    X = np.c_[t, np.ones_like(t)]
    beta = np.linalg.lstsq(X, x, rcond=None)[0]  # [2, F]
    trend = X @ beta
    return x - trend
