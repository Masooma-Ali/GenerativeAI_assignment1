"""Training loop for the Task 2 corruption classifier (balanced batches, cross-entropy)."""
import json, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from .data.pets import BalancedBatchSampler
from .models.classifier import CorruptionClassifier
from .tracking import mlflow_run


def build_clf(cfg):
    return CorruptionClassifier(cfg["channels"], cfg["dropout"])


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); ys, ps, loss_sum, n = [], [], 0.0, 0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, _, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += ce(logits, y).item(); n += y.numel()
        ys += y.cpu().tolist(); ps += logits.argmax(1).cpu().tolist()
    return np.array(ys), np.array(ps), loss_sum / n


def run_training_clf(cfg, train_ds, val_ds, out_dir, device, run_name="clf", experiment="task2_classifier",
                     trial=None, log_mlflow=True, num_workers=2, mlruns_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg.get("seed", 42)); np.random.seed(cfg.get("seed", 42))
    model = build_clf(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    ce = nn.CrossEntropyLoss()
    sampler = BalancedBatchSampler(len(train_ds) // 4, cfg["batch_size"])      # exactly B/4 per class per batch
    train_dl = DataLoader(train_ds, batch_sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=num_workers)
    best, best_p, history = -1.0, os.path.join(out_dir, "best.pt"), []

    with mlflow_run(log_mlflow, experiment, run_name, mlruns_dir or out_dir):
        if log_mlflow:
            import mlflow
            mlflow.log_params(cfg); mlflow.log_param("n_params", sum(p.numel() for p in model.parameters()))
        for ep in range(cfg["epochs"]):
            model.train(); t0 = time.time(); tl, correct, seen = 0.0, 0, 0
            for x, _, y in train_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x); loss = ce(logits, y)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                tl += loss.item() * y.numel(); correct += (logits.argmax(1) == y).sum().item(); seen += y.numel()
            sched.step()
            yv, pv, vloss = predict(model, val_dl, device)
            acc = float((yv == pv).mean()); f1 = float(f1_score(yv, pv, average="macro"))
            rec = {"epoch": ep, "train_loss": tl / seen, "train_acc": correct / seen,
                   "val_loss": vloss, "val_acc": acc, "val_macro_f1": f1, "lr": sched.get_last_lr()[0]}
            history.append(rec)
            print(f"ep {ep:3d} | train loss {rec['train_loss']:.4f} acc {rec['train_acc']:.3f} | "
                  f"val loss {vloss:.4f} acc {acc:.3f} macroF1 {f1:.3f} | {time.time()-t0:.0f}s")
            if f1 > best:
                best = f1
                torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": ep, "best": best}, best_p)
            if log_mlflow:
                mlflow.log_metrics({k: v for k, v in rec.items() if k != "epoch"}, step=ep)
            if trial is not None:
                trial.report(f1, ep)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
        if log_mlflow:
            mlflow.log_metric("best_val_macro_f1", best); mlflow.log_artifact(best_p)
        json.dump(history, open(os.path.join(out_dir, "history.json"), "w"))
    return best, history
