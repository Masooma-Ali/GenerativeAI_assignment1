"""Optuna study for the Task 2 corruption classifier (objective: validation macro-F1).

    python scripts/optuna_clf.py --data-root /kaggle/working/data --out /kaggle/working/optuna_clf \
        --n-trials 20 --epochs 8 --timeout 4800
Resumable (SQLite). --n-trials 0 regenerates the report only."""
import argparse, gc, os, sys
import optuna, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._common import load_pets
from src.data.pets import BalancedCorruptionDataset, PetRestorationDataset
from src.optuna_utils import make_report
from src.train_clf import run_training_clf

SPACE = {
    "lr":           "log-uniform [3e-4, 5e-3]",
    "batch_size":   "categorical {32, 64, 128}   (multiple of 4: balanced batches)",
    "channels":     "categorical {small [16,32,64,128], medium [32,64,128,256], large [64,128,256,512]}",
    "dropout":      "uniform [0.0, 0.5]",
    "weight_decay": "log-uniform [1e-6, 1e-2]",
}

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="data"); ap.add_argument("--out", default="optuna_clf")
ap.add_argument("--n-trials", type=int, default=20); ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--timeout", type=int, default=None); ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

images, split, val_manifest, _ = load_pets(a.data_root)
train_ds = BalancedCorruptionDataset(images, split["train"])
val_ds = PetRestorationDataset(images, manifest=val_manifest)
device = "cuda" if torch.cuda.is_available() else "cpu"


def objective(trial):
    cfg = dict(lr=trial.suggest_float("lr", 3e-4, 5e-3, log=True),
               batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
               channels=trial.suggest_categorical("channels", ["small", "medium", "large"]),
               dropout=trial.suggest_float("dropout", 0.0, 0.5),
               weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
               epochs=a.epochs, seed=42)
    tdir = os.path.join(a.out, "trials", f"trial_{trial.number:03d}")
    try:
        best, _ = run_training_clf(cfg, train_ds, val_ds, tdir, device, run_name=f"trial_{trial.number:03d}",
                                   experiment="task2_classifier_optuna", trial=trial,
                                   num_workers=a.workers, mlruns_dir=a.out)
    finally:
        p = os.path.join(tdir, "best.pt")
        if os.path.exists(p): os.remove(p)
        gc.collect(); torch.cuda.empty_cache()
    return best


study = optuna.create_study(study_name="task2_classifier", storage=f"sqlite:///{os.path.abspath(os.path.join(a.out, 'optuna.db'))}",
                            load_if_exists=True, direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2))
if a.n_trials > 0:
    study.optimize(objective, n_trials=a.n_trials, timeout=a.timeout)
make_report(study, a.out, SPACE, metric_name="val_macro_f1")
