"""ONE shared Optuna search for the three Task 2 specialist autoencoders.

Each trial trains ONE model on the mixture of the three corruption types (salt_pepper, blur, occlusion;
clean images are excluded because clean inputs bypass the experts). The objective is the validation
score on those three types. The winning architecture is then trained three times, independently
(scripts/train_specialists.py), once per corruption type.

    python scripts/optuna_specialists.py --data-root /kaggle/working/data --out /kaggle/working/optuna_spec \
        --n-trials 20 --epochs 12 --timeout 5400
Resumable (SQLite). --n-trials 0 regenerates the report only."""
import argparse, gc, os, sys
import optuna, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._common import load_pets
from src.data.pets import PetRestorationDataset
from src.optuna_utils import make_report
from src.train_ae import run_training

TYPES = ["salt_pepper", "blur", "occlusion"]
SPACE = {
    "lr":         "log-uniform [1e-4, 3e-3]",
    "batch_size": "categorical {32, 64, 128}",
    "latent_ch":  "categorical {16, 32, 64}      (bottleneck = latent_ch x 8 x 8 = 1024..4096 values)",
    "base":       "categorical {32, 48, 64}      (channel configuration: stages = base*[1,2,4,8])",
    "alpha":      "uniform [0.5, 0.95]           (L1 weight; SSIM weight = 1-alpha)",
    "(fixed)":    "dropout 0.0, weight_decay 1e-4, SpatialAE (as Task 1)",
}

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="data"); ap.add_argument("--out", default="optuna_spec")
ap.add_argument("--n-trials", type=int, default=20); ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--timeout", type=int, default=None); ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

images, split, val_manifest, _ = load_pets(a.data_root)
train_ds = PetRestorationDataset(images, indices=split["train"], types=TYPES)
val_ds = PetRestorationDataset(images, manifest=val_manifest, types=TYPES)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("train images:", len(train_ds), "| val samples (3 types):", len(val_ds))


def objective(trial):
    cfg = dict(lr=trial.suggest_float("lr", 1e-4, 3e-3, log=True),
               batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
               latent_ch=trial.suggest_categorical("latent_ch", [16, 32, 64]),
               base=trial.suggest_categorical("base", [32, 48, 64]),
               alpha=trial.suggest_float("alpha", 0.5, 0.95),
               dropout=0.0, weight_decay=1e-4, latent_mode="spatial", epochs=a.epochs, seed=42)
    tdir = os.path.join(a.out, "trials", f"trial_{trial.number:03d}")
    try:
        best, _ = run_training(cfg, train_ds, val_ds, tdir, device, run_name=f"trial_{trial.number:03d}",
                               experiment="task2_specialists_optuna", trial=trial,
                               num_workers=a.workers, mlruns_dir=a.out)
    finally:
        for f in ("last.pt", "best.pt"):
            p = os.path.join(tdir, f)
            if os.path.exists(p): os.remove(p)
        gc.collect(); torch.cuda.empty_cache()
    return best


study = optuna.create_study(study_name="task2_specialists", storage=f"sqlite:///{os.path.abspath(os.path.join(a.out, 'optuna.db'))}",
                            load_if_exists=True, direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
if a.n_trials > 0:
    study.optimize(objective, n_trials=a.n_trials, timeout=a.timeout)
make_report(study, a.out, SPACE, extra_cfg=dict(dropout=0.0, weight_decay=1e-4, latent_mode="spatial"),
            metric_name="val_score")
