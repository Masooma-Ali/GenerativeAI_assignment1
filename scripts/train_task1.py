"""Train the Task 1 universal autoencoder.
    python scripts/train_task1.py --data-root /kaggle/working/data --out /kaggle/working/task1 --epochs 2   # smoke test
    python scripts/train_task1.py --data-root /kaggle/working/data --out /kaggle/working/task1 --epochs 60
"""
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.pets import PetRestorationDataset, load_manifest
from src.train_ae import run_training

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="data"); ap.add_argument("--out", default="task1")
ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch-size", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--base", type=int, default=32)
ap.add_argument("--latent-dim", type=int, default=256); ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--alpha", type=float, default=0.8); ap.add_argument("--weight-decay", type=float, default=1e-4)
ap.add_argument("--config", default=None, help="JSON (e.g. Optuna best_config.json) overriding the hyper-parameters")
ap.add_argument("--resume", action="store_true"); ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args()

proc = os.path.join(a.data_root, "processed")
images = np.load(os.path.join(proc, "trainval_128.npy"))
split = json.load(open(os.path.join(proc, "split.json")))
val_manifest = load_manifest(os.path.join(proc, "val_manifest.jsonl"))
train_ds = PetRestorationDataset(images, indices=split["train"])
val_ds = PetRestorationDataset(images, manifest=val_manifest)

cfg = dict(base=a.base, latent_dim=a.latent_dim, dropout=a.dropout, alpha=a.alpha, lr=a.lr,
           batch_size=a.batch_size, epochs=a.epochs, weight_decay=a.weight_decay, seed=42)
if a.config:
    cfg.update({k: v for k, v in json.load(open(a.config)).items() if k in cfg})
    print("config overridden from", a.config)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| train:", len(train_ds), "val:", len(val_ds))
best, _ = run_training(cfg, train_ds, val_ds, a.out, device, resume=a.resume, num_workers=a.workers)
print("best val score:", best)
