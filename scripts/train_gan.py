"""Retrain the selected cGAN configuration for the full schedule.
    python scripts/train_gan.py --fs2k /kaggle/working/fs2k --config /kaggle/working/optuna_gan/best_config.json \
        --out /kaggle/working/gan_final --epochs 150           (add --resume to continue after a cut-off)"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.fs2k import FS2KDataset
from src.train_gan import run_training_gan

ap = argparse.ArgumentParser()
ap.add_argument("--fs2k", required=True); ap.add_argument("--out", default="gan_final")
ap.add_argument("--config", default=None); ap.add_argument("--epochs", type=int, default=150)
ap.add_argument("--resume", action="store_true"); ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--sample-every", type=int, default=5)
a = ap.parse_args()
# brief's starting point: lambda_l1 = 100
cfg = dict(lr_g=2e-4, lr_d=2e-4, batch_size=16, base=64, dropout=0.3, style_dim=16, lambda_l1=100.0, seed=42)
if a.config:
    cfg.update({k: v for k, v in json.load(open(a.config)).items() if k in cfg}); print("config overridden from", a.config)
cfg["epochs"] = a.epochs
train_ds = FS2KDataset(os.path.join(a.fs2k, "train.npz"), augment=True)
val_ds = FS2KDataset(os.path.join(a.fs2k, "val.npz"))
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| train", len(train_ds), "val", len(val_ds), "| cfg", cfg)
best, _ = run_training_gan(cfg, train_ds, val_ds, a.out, device, resume=a.resume, num_workers=a.workers, sample_every=a.sample_every)
print("best val score (SSIM - L1):", best)
