"""Reusable training loop for restoration autoencoders (Task 1 now; the same loop
trains the Task 2 specialists and is called by Optuna)."""
import contextlib, json, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from .losses import RestorationLoss, batch_metrics, combined_score
from .models.autoencoder import SpatialAE, UniversalAE


def build_model(cfg):
    if cfg.get("latent_mode", "flat") == "spatial":
        return SpatialAE(cfg["base"], cfg["latent_ch"], cfg["dropout"])
    return UniversalAE(cfg["base"], cfg["latent_dim"], cfg["dropout"])


@torch.no_grad()
def validate(model, loader, crit, device):
    model.eval()
    n = 0; tot = {"loss": 0., "psnr": 0., "ssim": 0.}
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss, _, _ = crit(pred, y)
        psnr, s = batch_metrics(pred, y)
        b = x.size(0); n += b
        tot["loss"] += loss.item() * b; tot["psnr"] += psnr.sum().item(); tot["ssim"] += s.sum().item()
    return {k: v / n for k, v in tot.items()}


def _sample_grid(model, val_ds, device, n=8):
    model.eval()
    xs, ys = zip(*[(val_ds[i][0], val_ds[i][1]) for i in range(n)])
    x, y = torch.stack(xs).to(device), torch.stack(ys).to(device)
    with torch.no_grad():
        pred = model(x)
    grid = make_grid(torch.cat([x, pred.clamp(0, 1), y]).cpu(), nrow=n)   # rows: input / restored / clean
    return (grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def run_training(cfg, train_ds, val_ds, out_dir, device, run_name="task1",
                 experiment="task1_universal_ae", trial=None, log_mlflow=True,
                 resume=False, num_workers=2, mlruns_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg.get("seed", 42)); np.random.seed(cfg.get("seed", 42))
    model = build_model(cfg).to(device)
    crit = RestorationLoss(cfg["alpha"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                          num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=num_workers)

    last_p, best_p = os.path.join(out_dir, "last.pt"), os.path.join(out_dir, "best.pt")
    start_ep, best = 0, -1.0
    if resume and os.path.exists(last_p):
        ck = torch.load(last_p, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start_ep, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {start_ep}")

    if log_mlflow:
        import mlflow
        from pathlib import Path
        base = os.path.abspath(mlruns_dir or out_dir)
        os.makedirs(base, exist_ok=True)
        # new MLflow versions disable the old file store, so we use SQLite
        mlflow.set_tracking_uri(f"sqlite:///{os.path.join(base, 'mlflow.db')}")
        if mlflow.get_experiment_by_name(experiment) is None:
            mlflow.create_experiment(experiment, artifact_location=Path(os.path.join(base, "artifacts")).as_uri())
        mlflow.set_experiment(experiment)
        run_ctx = mlflow.start_run(run_name=run_name, nested=mlflow.active_run() is not None)
    else:
        run_ctx = contextlib.nullcontext()

    history = []
    with run_ctx:
        if log_mlflow:
            mlflow.log_params({k: v for k, v in cfg.items()})
            mlflow.log_param("n_params", sum(p.numel() for p in model.parameters()))
        for ep in range(start_ep, cfg["epochs"]):
            model.train(); t0 = time.time(); run = {"loss": 0., "l1": 0., "ssim": 0.}; nb = 0
            for x, y, _ in train_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                loss, l1, s = crit(model(x), y)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                run["loss"] += loss.item(); run["l1"] += l1.item(); run["ssim"] += s.item(); nb += 1
            sched.step()
            v = validate(model, val_dl, crit, device)
            score = combined_score(v["psnr"], v["ssim"])
            rec = {"epoch": ep, "train_loss": run["loss"] / nb, "train_l1": run["l1"] / nb,
                   "train_ssim": run["ssim"] / nb, "val_loss": v["loss"], "val_psnr": v["psnr"],
                   "val_ssim": v["ssim"], "val_score": score, "lr": sched.get_last_lr()[0]}
            history.append(rec)
            print(f"ep {ep:3d} | train {rec['train_loss']:.4f} | val {v['loss']:.4f} "
                  f"PSNR {v['psnr']:.2f} SSIM {v['ssim']:.4f} score {score:.4f} | {time.time()-t0:.0f}s")

            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep, "best": best}, best_p)
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "cfg": cfg, "epoch": ep, "best": best}, last_p)

            if log_mlflow:
                mlflow.log_metrics({k: v_ for k, v_ in rec.items() if k != "epoch"}, step=ep)
                if ep % 5 == 0 or ep == cfg["epochs"] - 1:
                    mlflow.log_image(_sample_grid(model, val_ds, device), f"samples/epoch_{ep:03d}.png")
            if trial is not None:                      # Optuna pruning hook (used in the next step)
                trial.report(score, ep)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()

        if log_mlflow:
            mlflow.log_metric("best_val_score", best)
            mlflow.log_artifact(best_p)
        json.dump(history, open(os.path.join(out_dir, "history.json"), "w"))
    return best, history