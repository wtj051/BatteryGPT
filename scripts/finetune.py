#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import torch
import torch.nn as nn

from models.battery_gpt import BatteryGPT
from data.load_data import load_finetune_data  # must return a DataLoader, or (train_loader, val_loader[, test_loader])

from utils.metrics import evaluate_soh_rul
from utils.visualization import plot_training_curve


def parse_args():
    p = argparse.ArgumentParser(description="BatteryGPT Supervised Fine-tuning for SOH & RUL")
    p.add_argument("--config", default="configs/battery_gpt.yaml", help="Path to YAML config")
    p.add_argument("--dataset", default=None, help="Dataset key in config.data.partitions (e.g., NASA)")
    p.add_argument("--pretrained", default=None, help="Override path to pretrained checkpoint (.pth)")
    p.add_argument("--save", default=None, help="Override path to save the finetuned checkpoint (.pth)")
    p.add_argument("--device", default=None, help="'cuda' or 'cpu'")
    return p.parse_args()


def _as_tuple_loaders(obj):
    """
    Normalize output of load_finetune_data to (train_loader, val_loader, test_loader).
    Accepts: DataLoader | (train,) | (train,val) | (train,val,test)
    """
    if isinstance(obj, tuple) or isinstance(obj, list):
        if len(obj) == 1:
            return obj[0], None, None
        if len(obj) == 2:
            return obj[0], obj[1], None
        if len(obj) >= 3:
            return obj[0], obj[1], obj[2]
    # single loader
    return obj, None, None


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
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Finetune hyperparams (allow per-dataset overrides)
    ft_cfg = cfg.get("finetune", {})
    epochs = int(ft_cfg.get("epochs", 100))

    per_ds = ft_cfg.get("per_dataset", {})
    defaults = ft_cfg.get("defaults", {"lr": 1e-4, "batch_size": 32})
    lr = defaults.get("lr", 1e-4)
    batch_size = defaults.get("batch_size", 32)
    if dataset_key in per_ds:
        lr = float(per_ds[dataset_key].get("lr", lr))
        batch_size = int(per_ds[dataset_key].get("batch_size", batch_size))

    log_every = int(cfg.get("log_every_n_steps", 50))
    num_workers = int(cfg.get("num_workers", 4))

    # Model parameters
    mcfg = cfg.get("model", {})
    model = BatteryGPT(
        input_dims=mcfg.get("input_dims", [10, 10, 10]),
        hidden_dim=mcfg.get("hidden_dim", 128),
        n_heads=mcfg.get("n_heads", 8),
        short_win=mcfg.get("short_win", 5),
        long_win=mcfg.get("long_win", 50),
    ).to(device)

    # ----------------
    # Load pretrained weights
    # ----------------
    default_pretrained = os.path.join(save_dir, f"battery_gpt_{dataset_key}_pretrain.pth")
    pretrained_path = args.pretrained or default_pretrained
    if os.path.exists(pretrained_path):
        state = torch.load(pretrained_path, map_location="cpu")
        # support both {weights} and {"state_dict": weights}
        model.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state, strict=False)
        print(f"[Loaded] Pretrained weights from: {pretrained_path}")
    else:
        print(f"[Warning] Pretrained checkpoint not found at {pretrained_path}. Proceeding without it.")

    # ----------------
    # Optimizer & Loss
    # ----------------
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=cfg.get("optimizer", {}).get("weight_decay", 0.0),
    )
    criterion = nn.MSELoss()

    # ----------------
    # Data loaders
    # Expected signature:
    #   load_finetune_data(data_root, dataset_key, batch_size, num_workers)
    # Should return DataLoader or tuple of loaders
    # ----------------
    data_root = cfg.get("data", {}).get("root", "data")
    loaders = load_finetune_data(
        data_root=data_root,
        dataset_key=dataset_key,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    train_loader, val_loader, test_loader = _as_tuple_loaders(loaders)

    # ----------------
    # Training loop
    # ----------------
    history = {"train_loss": [], "val_loss": []}
    global_step = 0
    grad_clip = float(cfg.get("train", {}).get("gradient_clip_norm", 1.0))

    for epoch in range(1, epochs + 1):
        model.train()
        running, nb = 0.0, 0

        for batch in train_loader:
            # batch is expected as (inputs, soh_labels, rul_labels)
            inputs, soh_labels, rul_labels = batch
            inputs = [x.to(device) for x in inputs]
            soh_labels = soh_labels.to(device).float()
            rul_labels = rul_labels.to(device).float()

            soh_pred, rul_pred = model(inputs)  # [B,1], [B,1]

            loss_soh = criterion(soh_pred.squeeze(-1), soh_labels)
            loss_rul = criterion(rul_pred.squeeze(-1), rul_labels)
            loss = loss_soh + loss_rul

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running += float(loss.item())
            nb += 1
            global_step += 1

            if log_every and (global_step % log_every == 0):
                print(
                    f"[Finetune] epoch {epoch} step {global_step} | "
                    f"loss={loss.item():.6f} (soh={loss_soh.item():.6f}, rul={loss_rul.item():.6f})"
                )

        epoch_train_loss = running / max(1, nb)
        history["train_loss"].append(epoch_train_loss)

        # ------------- optional validation -------------
        if val_loader is not None:
            model.eval()
            v_running, v_nb = 0.0, 0
            with torch.no_grad():
                for vbatch in val_loader:
                    v_inputs, v_soh, v_rul = vbatch
                    v_inputs = [x.to(device) for x in v_inputs]
                    v_soh = v_soh.to(device).float()
                    v_rul = v_rul.to(device).float()

                    v_soh_pred, v_rul_pred = model(v_inputs)
                    v_loss = criterion(v_soh_pred.squeeze(-1), v_soh) + criterion(v_rul_pred.squeeze(-1), v_rul)
                    v_running += float(v_loss.item())
                    v_nb += 1
            epoch_val_loss = v_running / max(1, v_nb)
            history["val_loss"].append(epoch_val_loss)
            print(f"Epoch {epoch}/{epochs} - train_loss: {epoch_train_loss:.6f} | val_loss: {epoch_val_loss:.6f}")
        else:
            print(f"Epoch {epoch}/{epochs} - train_loss: {epoch_train_loss:.6f}")

    # ----------------
    # Plot training curve
    # ----------------
    plot_training_curve(history, out_path=os.path.join(results_dir, "curves", "finetune_loss.png"),
                        title="Fine-tuning Loss")

    # ----------------
    # Optional quick test evaluation (if provided by loader)
    # ----------------
    if test_loader is not None:
        model.eval()
        all_soh_t, all_soh_p = [], []
        all_rul_t, all_rul_p = [], []
        with torch.no_grad():
            for tbatch in test_loader:
                t_inputs, t_soh, t_rul = tbatch
                t_inputs = [x.to(device) for x in t_inputs]
                t_soh = t_soh.to(device).float()
                t_rul = t_rul.to(device).float()

                t_soh_pred, t_rul_pred = model(t_inputs)
                all_soh_t.append(t_soh.cpu());  all_soh_p.append(t_soh_pred.squeeze(-1).cpu())
                all_rul_t.append(t_rul.cpu());  all_rul_p.append(t_rul_pred.squeeze(-1).cpu())

        soh_true = torch.cat(all_soh_t); soh_pred = torch.cat(all_soh_p)
        rul_true = torch.cat(all_rul_t); rul_pred = torch.cat(all_rul_p)
        metrics = evaluate_soh_rul(soh_true, soh_pred, rul_true, rul_pred)
        print("[Test metrics]", {k: f"{v:.6f}" for k, v in metrics.items()})

    # ----------------
    # Save checkpoint
    # ----------------
    save_path = args.save or os.path.join(save_dir, f"battery_gpt_{dataset_key}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"[Saved] Finetuned weights -> {save_path}")


if __name__ == "__main__":
    main()
