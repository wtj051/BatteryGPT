import numpy as np
from typing import Dict, Sequence

def windowed_view(x: np.ndarray, win: int, step: int) -> np.ndarray:
    """
    Create sliding windows over time axis.
    x: [T, F] -> [Nw, win, F]
    """
    T, F = x.shape
    if T < win:
        return np.zeros((0, win, F))
    idx = np.arange(0, T - win + 1, step)
    return np.stack([x[i:i+win] for i in idx], axis=0)

def stat_features(win_x: np.ndarray) -> Dict[str, np.ndarray]:
    """
    win_x: [Nw, win, F]
    Returns statistical window features: mean, std, min, max, skewness, kurtosis.
    """
    eps = 1e-8
    mean = win_x.mean(axis=1)
    std = win_x.std(axis=1) + eps
    minv = win_x.min(axis=1)
    maxv = win_x.max(axis=1)
    z = (win_x - mean[:, None, :]) / std[:, None, :]
    skew = np.mean(z**3, axis=1)
    kurt = np.mean(z**4, axis=1) - 3.0
    return {"mean": mean, "std": std, "min": minv, "max": maxv, "skew": skew, "kurt": kurt}

def freq_features(win_x: np.ndarray, topk: int = 3) -> Dict[str, np.ndarray]:
    """
    Simple frequency-domain features using rFFT:
    Returns top-k peak frequencies and powers per feature.
    """
    Nw, W, F = win_x.shape
    if Nw == 0:
        return {"top_freqs": np.zeros((0, topk, F)), "top_power": np.zeros((0, topk, F))}
    fft = np.fft.rfft(win_x, axis=1)               # [Nw, W/2+1, F]
    power = (fft.real**2 + fft.imag**2)            # [Nw, W/2+1, F]
    freqs = np.fft.rfftfreq(W, d=1.0)              # [W/2+1]
    idx_top = np.argsort(power, axis=1)[:, -topk:, :]  # [Nw, topk, F]
    top_freqs = np.take_along_axis(np.tile(freqs[None, :, None], (Nw, 1, F)), idx_top, axis=1)
    top_power = np.take_along_axis(power, idx_top, axis=1)
    return {"top_freqs": top_freqs, "top_power": top_power}

def icc_features(voltage: Sequence[float], capacity: Sequence[float], bins: int = 64) -> Dict[str, np.ndarray]:
    """
    Incremental Capacity Curve proxy features for a single cycle:
    voltage: [T,]  capacity: [T,]
    Returns histogram of dQ/dV and peak/mean/std statistics.
    """
    voltage = np.asarray(voltage)
    capacity = np.asarray(capacity)
    dV = np.diff(voltage) + 1e-8
    dQ = np.diff(capacity)
    dQdV = dQ / dV
    hist, edges = np.histogram(dQdV, bins=bins, density=True)
    peak = edges[1:][np.argmax(hist)] if hist.size else 0.0
    return {
        "icc_hist": hist,
        "icc_peak": np.array([peak]),
        "icc_mean": np.array([dQdV.mean() if dQdV.size else 0.0]),
        "icc_std":  np.array([dQdV.std()  if dQdV.size else 0.0])
    }

class FeatureExtractor:
    """
    High-level feature pipeline (optional) for classical ML or analysis.
    Combines windowed statistical and frequency features.
    """
    def __init__(self, win: int = 50, step: int = 10, topk: int = 3):
        self.win = win
        self.step = step
        self.topk = topk

    def extract(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        x: [T, F] -> Dict of features (window statistics + top-k frequency peaks).
        """
        windows = windowed_view(x, self.win, self.step)
        out = {}
        out.update(stat_features(windows))
        out.update(freq_features(windows, self.topk))
        return out
