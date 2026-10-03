"""Train the final Task 2 classifier.
    python scripts/train_clf.py --data-root /kaggle/working/data --out /kaggle/working/clf_final \
        --config /kaggle/working/optuna_clf/best_config.json --epochs 30"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._common import load_pets
from src.data.pets import BalancedCorruptionDataset, PetRestorationDataset
from src.train_clf import run_training_clf

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="data"); ap.add_argument("--out", default="clf_final")
ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--channels", default="medium")
ap.add_argument("--dropout", type=float, default=0.3); ap.add_argument("--weight-decay", type=float, default=1e-4)
ap.add_argument("--config", default=None); ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args()

images, split, val_manifest, _ = load_pets(a.data_root)
train_ds = BalancedCorruptionDataset(images, split["train"])
val_ds = PetRestorationDataset(images, manifest=val_manifest)
cfg = dict(lr=a.lr, batch_size=a.batch_size, channels=a.channels, dropout=a.dropout,
           weight_decay=a.weight_decay, epochs=a.epochs, seed=42)
if a.config:
    cfg.update({k: v for k, v in json.load(open(a.config)).items() if k in cfg and k != "epochs"})
    print("config overridden from", a.config)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| cfg:", cfg)
best, _ = run_training_clf(cfg, train_ds, val_ds, a.out, device, num_workers=a.workers)
print("best val macro-F1:", best)
