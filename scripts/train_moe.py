"""Train the final soft MoE with the Optuna-selected hyper-parameters.
    python scripts/train_moe.py --data-root D --clf-ckpt clf_final/best.pt --spec-dir specialists \
        --config optuna_moe/best_config.json --out moe_final --epochs 24 --warmup 3"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._common import load_pets
from src.data.pets import BalancedCorruptionDataset, PetRestorationDataset
from src.train_moe import run_training_moe

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="data"); ap.add_argument("--clf-ckpt", required=True)
ap.add_argument("--spec-dir", required=True); ap.add_argument("--config", default=None)
ap.add_argument("--out", default="moe_final"); ap.add_argument("--epochs", type=int, default=24)
ap.add_argument("--warmup", type=int, default=3); ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args()

# brief's suggested starting point: alpha 0.8 (lambda_s 0.2), lambda_c 0.1, lambda_b 0.01
cfg = dict(batch_size=32, lr_warm=3e-4, weight_decay=1e-4, epoch_fraction=0.25, seed=42,
           lr=1e-4, temperature=1.0, lambda_c=0.1, lambda_b=0.01, alpha=0.8)
if a.config:
    cfg.update({k: v for k, v in json.load(open(a.config)).items()}); print("config overridden from", a.config)
cfg.update(epochs=a.epochs, warmup_epochs=a.warmup)

images, split, val_manifest, _ = load_pets(a.data_root)
train_ds = BalancedCorruptionDataset(images, split["train"])
val_ds = PetRestorationDataset(images, manifest=val_manifest)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "| cfg:", cfg)
best, _ = run_training_moe(cfg, train_ds, val_ds, a.clf_ckpt, a.spec_dir, a.out, device, num_workers=a.workers)
print("best val score:", best)
