"""Optuna hyper-parameter study for the Task 1 universal autoencoder.

    python scripts/optuna_task1.py --data-root /kaggle/working/data --out /kaggle/working/optuna_task1 \
        --n-trials 24 --epochs 12 --timeout 5400

* Objective (maximise): val_score = 0.5*SSIM + 0.5*min(PSNR,40)/40 on the fixed validation manifest
  (independent of alpha, so trials with different loss weights are compared fairly).
* Study is stored in SQLite -> re-run the same command to RESUME after a session cut-off.
* Bad trials are stopped early by a MedianPruner.
* Every trial is logged to MLflow (experiment "task1_optuna").
* Use --n-trials 0 to only regenerate the report from an existing study.
"""
import argparse, gc, json, os, shutil, sys
import numpy as np
import optuna
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.pets import PetRestorationDataset, load_manifest
from src.train_ae import run_training

SPACE = {   # documented for the report
    "lr":         "log-uniform [3e-4, 3e-3]",
    "batch_size": "categorical {32, 64, 128}",
    "latent_dim": "categorical {256, 512, 1024, 2048, 4096}  (bottleneck dimension)",
    "base":       "categorical {16, 32, 48}            (encoder base channels; stages = base*[1,2,4,8,8])",
    "dropout":    "uniform [0.0, 0.3]                  (on latent code)",
    "alpha":      "uniform [0.5, 0.95]                 (L1 weight; SSIM weight = 1-alpha)",
}


def suggest(trial):
    return dict(
        lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
        latent_dim=trial.suggest_categorical("latent_dim", [256, 512, 1024, 2048, 4096]),
        base=trial.suggest_categorical("base", [16, 32, 48]),
        dropout=trial.suggest_float("dropout", 0.0, 0.3),
        alpha=trial.suggest_float("alpha", 0.5, 0.95),
    )


def make_report(study, out):
    df = study.trials_dataframe()
    df.to_csv(os.path.join(out, "trials.csv"), index=False)
    states = df["state"].value_counts().to_dict()
    comp = df[df.state == "COMPLETE"]
    lines = ["Optuna study: task1_universal_ae", f"trials by state: {states}", "", "Search space:"]
    lines += [f"  {k:11s} {v}" for k, v in SPACE.items()]
    if len(comp):
        bt = study.best_trial
        lines += ["", f"Best trial: #{bt.number}  val_score={bt.value:.4f}", "Best params:"]
        lines += [f"  {k}: {v}" for k, v in bt.params.items()]
        try:
            imp = optuna.importance.get_param_importances(study)
            lines += ["", "Parameter importance (fANOVA):"] + [f"  {k}: {v:.3f}" for k, v in imp.items()]
        except Exception as e:
            imp = None; lines += ["", f"(importance not computed: {e})"]
        # history plot
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.scatter(comp.number, comp.value, label="completed trial")
        ax.plot(comp.number, comp.value.cummax(), "r-", label="best so far")
        ax.set_xlabel("trial"); ax.set_ylabel("val score"); ax.legend(); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(out, "optuna_history.png"), dpi=140); plt.close(fig)
        # parameter vs score
        pc = [c for c in comp.columns if c.startswith("params_")]
        fig, axs = plt.subplots(2, 3, figsize=(11, 6))
        for ax, c in zip(axs.ravel(), pc):
            ax.scatter(comp[c], comp.value, s=18); ax.set_xlabel(c.replace("params_", "")); ax.set_ylabel("val score")
            if c == "params_lr": ax.set_xscale("log")
            ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(out, "optuna_params.png"), dpi=140); plt.close(fig)
        if imp:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(list(imp)[::-1], list(imp.values())[::-1]); ax.set_xlabel("importance")
            fig.tight_layout(); fig.savefig(os.path.join(out, "optuna_importance.png"), dpi=140); plt.close(fig)
        best_cfg = dict(bt.params, weight_decay=1e-4)
        json.dump(best_cfg, open(os.path.join(out, "best_config.json"), "w"), indent=2)
    txt = "\n".join(lines); open(os.path.join(out, "summary.txt"), "w").write(txt); print("\n" + txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data"); ap.add_argument("--out", default="optuna_task1")
    ap.add_argument("--n-trials", type=int, default=24); ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=None, help="stop starting new trials after N seconds")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    proc = os.path.join(a.data_root, "processed")
    images = np.load(os.path.join(proc, "trainval_128.npy"))
    split = json.load(open(os.path.join(proc, "split.json")))
    train_ds = PetRestorationDataset(images, indices=split["train"])
    val_ds = PetRestorationDataset(images, manifest=load_manifest(os.path.join(proc, "val_manifest.jsonl")))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def objective(trial):
        cfg = suggest(trial); cfg.update(epochs=a.epochs, weight_decay=1e-4, seed=42)
        tdir = os.path.join(a.out, "trials", f"trial_{trial.number:03d}")
        try:
            best, _ = run_training(cfg, train_ds, val_ds, tdir, device, run_name=f"trial_{trial.number:03d}",
                                   experiment="task1_optuna", trial=trial, num_workers=a.workers, mlruns_dir=a.out)
        finally:                                   # keep disk usage small
            for f in ("last.pt", "best.pt"):
                p = os.path.join(tdir, f)
                if os.path.exists(p): os.remove(p)
            gc.collect(); torch.cuda.empty_cache()
        return best

    study = optuna.create_study(
        study_name="task1_universal_ae", storage=f"sqlite:///{os.path.abspath(os.path.join(a.out, 'optuna.db'))}",
        load_if_exists=True, direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
    if a.n_trials > 0:
        study.optimize(objective, n_trials=a.n_trials, timeout=a.timeout)
    make_report(study, a.out)


if __name__ == "__main__":
    main()
