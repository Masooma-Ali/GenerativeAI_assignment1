"""Optuna study for the Task 3 soft MoE: joint fine-tuning lr, temperature, classification weight,
balance weight and the L1/SSIM reconstruction weighting. Trials whose routing collapses
(one branch > 60% average weight, or any branch < 5%) are pruned.

    python scripts/optuna_moe.py --data-root D --clf-ckpt clf_final/best.pt --spec-dir specialists --out optuna_moe \
        --n-trials 15 --epochs 6 --timeout 7200"""
import argparse, gc, os, sys
import optuna, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts._common import load_pets
from src.data.pets import BalancedCorruptionDataset, PetRestorationDataset
from src.optuna_utils import make_report
from src.train_moe import run_training_moe

FIXED = dict(batch_size=32, lr_warm=3e-4, warmup_epochs=2, weight_decay=1e-4, epoch_fraction=0.25, seed=42)
SPACE = {
    "lr":          "log-uniform [2e-5, 3e-4]   (joint fine-tuning; smaller than the specialists' 6.5e-4)",
    "temperature": "uniform [0.5, 2.0]",
    "lambda_c":    "log-uniform [0.01, 1.0]    (classification / CE weight; brief suggests 0.1)",
    "lambda_b":    "log-uniform [1e-3, 1.0]    (balance weight; brief suggests 0.01)",
    "alpha":       "uniform [0.6, 0.95]        (L1 weight; SSIM weight = 1-alpha; brief suggests 0.8)",
    "(fixed)":     f"{FIXED}",
}
ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="data"); ap.add_argument("--clf-ckpt", required=True)
ap.add_argument("--spec-dir", required=True); ap.add_argument("--out", default="optuna_moe")
ap.add_argument("--n-trials", type=int, default=15); ap.add_argument("--epochs", type=int, default=6)
ap.add_argument("--timeout", type=int, default=None); ap.add_argument("--workers", type=int, default=2)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

images, split, val_manifest, _ = load_pets(a.data_root)
train_ds = BalancedCorruptionDataset(images, split["train"])
val_ds = PetRestorationDataset(images, manifest=val_manifest)
device = "cuda" if torch.cuda.is_available() else "cpu"


def objective(trial):
    cfg = dict(FIXED, lr=trial.suggest_float("lr", 2e-5, 3e-4, log=True),
               temperature=trial.suggest_float("temperature", 0.5, 2.0),
               lambda_c=trial.suggest_float("lambda_c", 0.01, 1.0, log=True),
               lambda_b=trial.suggest_float("lambda_b", 1e-3, 1.0, log=True),
               alpha=trial.suggest_float("alpha", 0.6, 0.95), epochs=a.epochs)
    tdir = os.path.join(a.out, "trials", f"trial_{trial.number:03d}")
    try:
        best, _ = run_training_moe(cfg, train_ds, val_ds, a.clf_ckpt, a.spec_dir, tdir, device,
                                   run_name=f"trial_{trial.number:03d}", experiment="task3_moe_optuna",
                                   trial=trial, num_workers=a.workers, mlruns_dir=a.out)
    finally:
        p = os.path.join(tdir, "best.pt")
        if os.path.exists(p): os.remove(p)
        gc.collect(); torch.cuda.empty_cache()
    if best < 0:
        raise optuna.TrialPruned()                  # never produced a valid (non-collapsed) joint epoch
    return best


study = optuna.create_study(study_name="task3_moe", storage=f"sqlite:///{os.path.abspath(os.path.join(a.out, 'optuna.db'))}",
                            load_if_exists=True, direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
                            pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1))
if a.n_trials > 0:
    study.optimize(objective, n_trials=a.n_trials, timeout=a.timeout)
make_report(study, a.out, SPACE, extra_cfg=dict(FIXED), metric_name="val_score")
