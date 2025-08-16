import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Sequence, Optional, Dict

def _ensure_dir(path: str):
    if path is None:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def plot_training_curve(history: Dict[str, Sequence[float]], out_path: Optional[str] = None, title: str = "Training Curve"):
    """
    history: e.g., {"train_loss":[...], "val_loss":[...], "metric":[...]}
    """
    plt.figure()
    for k, v in history.items():
        plt.plot(np.arange(1, len(v) + 1), v, label=k)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    if out_path:
        _ensure_dir(out_path)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()

def plot_pred_vs_true(y_true: Sequence[float], y_pred: Sequence[float], out_path: Optional[str] = None, title: str = "Prediction vs Ground Truth"):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    lim_min = min(y_true.min(), y_pred.min()) if y_true.size and y_pred.size else 0.0
    lim_max = max(y_true.max(), y_pred.max()) if y_true.size and y_pred.size else 1.0
    plt.figure()
    plt.scatter(y_true, y_pred, s=10, alpha=0.6)
    plt.plot([lim_min, lim_max], [lim_min, lim_max], linestyle="--")
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.title(title)
    if out_path:
        _ensure_dir(out_path)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()

def plot_error_hist(errors: Sequence[float], out_path: Optional[str] = None, bins: int = 30, title: str = "Error Distribution"):
    errors = np.asarray(errors)
    plt.figure()
    plt.hist(errors, bins=bins, alpha=0.8)
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.title(title)
    if out_path:
        _ensure_dir(out_path)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()

def plot_feature_importance(names: Sequence[str], importances: Sequence[float], out_path: Optional[str] = None, title: str = "Feature Importance"):
    names = list(names); imps = np.asarray(importances)
    order = np.argsort(-imps)
    names = [names[i] for i in order]
    imps = imps[order]
    plt.figure(figsize=(6, max(2, 0.28 * len(names))))
    y = np.arange(len(names))
    plt.barh(y, imps)
    plt.yticks(y, names)
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.title(title)
    if out_path:
        _ensure_dir(out_path)
        plt.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()
