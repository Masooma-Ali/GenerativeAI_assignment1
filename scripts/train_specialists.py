"""Train the three Task 2 specialist autoencoders INDEPENDENTLY (same architecture, separate weights,
separate random seeds). Each sees only its own corruption type; the target is always the clean image.

    python scripts/train_specialists.py --data-root /kaggle/working/data --out /kaggle/working/specialists \
        --config /kaggle/working/optuna_spec/best_config.json --epochs 60
    (--types blur   trains just one specialist)"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._common import load_pets
from src.data.pets import PetRestorationDataset
from src.train_ae import run_training

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="data"); ap.add_argument("--out", default="specialists")
ap.add_argument("--config", required=True); ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--types", nargs="+", default=["salt_pepper", "blur", "occlusion"])
ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args()

images, split, val_manifest, _ = load_pets(a.data_root)
base_cfg = json.load(open(a.config)); base_cfg.update(epochs=a.epochs, latent_mode="spatial")
device = "cuda" if torch.cuda.is_available() else "cpu"

for i, t in enumerate(["salt_pepper", "blur", "occlusion"]):
    if t not in a.types: continue
    cfg = dict(base_cfg, seed=42 + i, corruption=t)                  # different init per specialist
    train_ds = PetRestorationDataset(images, indices=split["train"], types=[t])
    val_ds = PetRestorationDataset(images, manifest=val_manifest, types=[t])
    print(f"\n=== specialist: {t} | train images {len(train_ds)} | val samples {len(val_ds)} | cfg {cfg}")
    best, _ = run_training(cfg, train_ds, val_ds, os.path.join(a.out, t), device,
                           run_name=f"specialist_{t}", experiment="task2_specialists",
                           num_workers=a.workers, mlruns_dir=a.out)
    print(f"[{t}] best val score: {best:.4f}")
