# BatteryGPT Checkpoints

This folder stores pre-trained and fine-tuned checkpoints.

We do not commit large binaries into the repository. Instead, we provide:
1) A machine-readable index (`index.json`) with filenames, sizes, SHA256 checksums, metrics, and download URLs.
2) Helper scripts to auto-download and verify checkpoints.

## Quick Start

```bash
# Option A: Auto-download the checkpoint for NASA before evaluation
python scripts/download_checkpoints.py --dataset NASA --stage finetune

# Option B: If you already downloaded a file, verify integrity
python scripts/verify_checkpoint.py --file checkpoints/battery_gpt_NASA_finetune.pth
