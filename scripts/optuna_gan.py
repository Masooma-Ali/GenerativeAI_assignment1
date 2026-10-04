"""Optuna study for the Task 4 cGAN (short runs; the winner is retrained for the full schedule).
Tunes: generator lr, discriminator lr, batch size, base channels, dropout, style-embedding dim, lambda_L1.
Objective (maximise): validation SSIM - validation L1 (the official test set is never touched).

    python scripts/optuna_gan.py --fs2k /kaggle/working/fs2k --out /kaggle/working/optuna_gan --n-trials 20 --epochs 20 --timeout 5400"""
import argparse, gc, os, sys
import optuna, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.fs2k import FS2KDataset
from src.optuna_utils import make_report
from src.train_gan import run_training_gan

SPACE = {
    "lr_g":       "log-uniform [5e-5, 1e-3]",
    "lr_d":       "log-uniform [5e-5, 1e-3]",
    "batch_size": "categorical {8, 16, 32}",
    "base":       "categorical {32, 48, 64}      (base channel count of G and D)",
    "dropout":    "uniform [0.0, 0.5]            (first three decoder blocks)",
    "style_dim":  "categorical {8, 16, 32, 64}   (style-embedding dimension)",
    "lambda_l1":  "log-uniform [10, 300]",
}
ap = argparse.ArgumentParser()
ap.add_argument("--fs2k", required=True); ap.add_argument("--out", default="optuna_gan")
ap.add_argument("--n-trials", type=int, default=20); ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--timeout", type=int, default=None); ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
train_ds = FS2KDataset(os.path.join(a.fs2k, "train.npz"), augment=True)
val_ds = FS2KDataset(os.path.join(a.fs2k, "val.npz"))
device = "cuda" if torch.cuda.is_available() else "cpu"


def objective(trial):
    cfg = dict(lr_g=trial.suggest_float("lr_g", 5e-5, 1e-3, log=True), lr_d=trial.suggest_float("lr_d", 5e-5, 1e-3, log=True),
               batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]), base=trial.suggest_categorical("base", [32, 48, 64]),
               dropout=trial.suggest_float("dropout", 0.0, 0.5), style_dim=trial.suggest_categorical("style_dim", [8, 16, 32, 64]),
               lambda_l1=trial.suggest_float("lambda_l1", 10, 300, log=True), epochs=a.epochs, seed=42)
    tdir = os.path.join(a.out, "trials", f"trial_{trial.number:03d}")
    try:
        best, _ = run_training_gan(cfg, train_ds, val_ds, tdir, device, run_name=f"trial_{trial.number:03d}",
                                   experiment="task4_cgan_optuna", trial=trial, num_workers=a.workers,
                                   mlruns_dir=a.out, sample_every=10_000)
    finally:
        for f in ("last.pt", "best.pt"):
            p = os.path.join(tdir, f)
            if os.path.exists(p): os.remove(p)
        gc.collect(); torch.cuda.empty_cache()
    return best


study = optuna.create_study(study_name="task4_cgan", storage=f"sqlite:///{os.path.abspath(os.path.join(a.out, 'optuna.db'))}",
                            load_if_exists=True, direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5))
if a.n_trials > 0:
    study.optimize(objective, n_trials=a.n_trials, timeout=a.timeout)
make_report(study, a.out, SPACE, metric_name="val_ssim_minus_l1")
