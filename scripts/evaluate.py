#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import torch
import torch.nn as nn

from models.battery_gpt import BatteryGPT
from data.load_data import load_test_data  # must return a DataLoader or (test_loader,)

from utils.metrics import evaluate_soh_rul
from utils.visualization import (
    plot_pred_vs_true,
    plot_error_hist,
)

def parse_args():
    p = argparse.ArgumentParser(description="BatteryGPT Evaluation (SOH & RUL)")
    p.add_argument("--config", default="configs/battery_gpt.yaml", help="Path to YAML config")
    p.add_argument("--dataset", default=None, help="Dataset key in config.data.partitions (e.g., NASA)")
    p.add_argument("--checkpoint", default=None, help="Override path to the finetuned checkpoint (.pth)")
    p.add_argument("--device", default=None, help="'cuda' or 'cpu'")
    p.add_argument("--save_csv", default=None, help="Optional CSV to append a row with metrics")
    return p.parse_args()


def _as_loader(obj):
    # Normalize output from load_test_data to a single DataLoader
    if isinstance(obj, (tuple, list)) and len(obj) >= 1:
        return obj[0]
    return obj


def main():
    args = parse_args()

    # ----------------
    # Load config
    # ----------------
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = args.device or cfg.get("device", "cuda")
    dataset_key = args.dataset or cfg.get("data", {}).get("dataset", "NASA")

    save_dir = cfg.get("save_dir", "checkpoints")
    results_dir = cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)

    num_workers = int(cfg.get("num_workers", 4))

    # ----------------
    # Build model
    # ----------------
    mcfg = cfg.get("model", {})
    model = BatteryGPT(
        input_dims=mcfg.get("input_dims", [10, 10, 10]),
        hidden_dim=mcfg.get("hidden_dim", 128),
        n_heads=mcfg.get("n_heads", 8),
        short_win=mcfg.get("short_win", 5),
        long_win=mcfg.get("long_win", 50),
    ).to(device)

    # ----------------
    # Load checkpoint
    # ----------------
    default_ckpt = os.path.join(save_dir, f"battery_gpt_{dataset_key}.pth")
    ckpt_path = args.checkpoint or default_ckpt
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            f"Use --checkpoint to specify a valid path."
        )

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state, strict=False)
    model.eval()
    print(f"[Loaded] Finetuned checkpoint: {ckpt_path}")

    # ----------------
    # Test loader
    # Expected signature (adapt as needed in your data.load_data):
    #   load_test_data(data_root, dataset_key, batch_size, num_workers) -> DataLoader or (DataLoader,)
    # Each batch: (inputs, soh_labels, rul_labels)
    # ----------------
    data_root = cfg.get("data", {}).get("root", "data")
    # batch size not critical for eval; default to per-dataset or 32
    ft_cfg = cfg.get("finetune", {})
    defaults = ft_cfg.get("defaults", {"batch_size": 32})
    per_ds = ft_cfg.get("per_dataset", {})
    batch_size = defaults.get("batch_size", 32)
    if dataset_key in per_ds:
        batch_size = int(per_ds[dataset_key].get("batch_size", batch_size))

    test_loader = _as_loader(
        load_test_data(
            data_root=data_root,
            dataset_key=dataset_key,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    )

    # ----------------
    # Inference
    # ----------------
    all_soh_true, all_soh_pred = [], []
    all_rul_true, all_rul_pred = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs, soh_t, rul_t = batch
            inputs = [x.to(device) for x in inputs]
            soh_t = soh_t.to(device).float()
            rul_t = rul_t.to(device).float()

            soh_p, rul_p = model(inputs)  # [B,1], [B,1]

            all_soh_true.append(soh_t.cpu())
            all_soh_pred.append(soh_p.squeeze(-1).cpu())
            all_rul_true.append(rul_t.cpu())
            all_rul_pred.append(rul_p.squeeze(-1).cpu())

    soh_true = torch.cat(all_soh_true).numpy()
    soh_pred = torch.cat(all_soh_pred).numpy()
    rul_true = torch.cat(all_rul_true).numpy()
    rul_pred = torch.cat(all_rul_pred).numpy()

    # ----------------
    # Metrics (MAE/RMSE/R2 + RUL_RE)
    # ----------------
    metrics = evaluate_soh_rul(soh_true, soh_pred, rul_true, rul_pred)
    pretty = " | ".join([f"{k}={v:.6f}" for k, v in metrics.items()])
    print(f"[{dataset_key}] {pretty}")

    # ----------------
    # Save metrics text and optional CSV
    # ----------------
    txt_path = os.path.join(results_dir, f"{dataset_key}_eval_results.txt")
    with open(txt_path, "w") as f:
        f.write(f"Checkpoint: {ckpt_path}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.6f}\n")
    print(f"[Saved] {txt_path}")

    if args.save_csv:
        import csv
        header = ["dataset", "checkpoint"] + list(metrics.keys())
        row = [dataset_key, os.path.basename(ckpt_path)] + [f"{metrics[k]:.6f}" for k in metrics.keys()]
        write_header = not os.path.exists(args.save_csv)
        with open(args.save_csv, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(header)
            w.writerow(row)
        print(f"[Appended] {args.save_csv}")

    # ----------------
    # Plots
    # ----------------
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_pred_vs_true(soh_true, soh_pred, out_path=os.path.join(plots_dir, f"{dataset_key}_soh_pred_vs_true.png"),
                      title=f"{dataset_key} SOH: Prediction vs Ground Truth")
    plot_pred_vs_true(rul_true, rul_pred, out_path=os.path.join(plots_dir, f"{dataset_key}_rul_pred_vs_true.png"),
                      title=f"{dataset_key} RUL: Prediction vs Ground Truth")
    plot_error_hist((soh_pred - soh_true), out_path=os.path.join(plots_dir, f"{dataset_key}_soh_error_hist.png"),
                    title=f"{dataset_key} SOH Error Distribution")
    plot_error_hist((rul_pred - rul_true), out_path=os.path.join(plots_dir, f"{dataset_key}_rul_error_hist.png"),
                    title=f"{dataset_key} RUL Error Distribution")


if __name__ == "__main__":
    main()
