"""Joint training of the soft MoE: gate warm-up (experts frozen) -> joint fine-tuning (smaller lr).

L = alpha*L1 + (1-alpha)*(1-SSIM) + lambda_c*CE(gate logits/T, true corruption) + lambda_b*sum_i (mean_w_i - 1/4)^2
The balance term is meaningful because training batches are class-balanced (see BalancedBatchSampler)."""
import json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader

from .data.corruptions import CLASSES
from .data.pets import BalancedBatchSampler
from .losses import batch_metrics, combined_score
from .models.moe import SoftMoE
from .tracking import mlflow_run
from .train_ae import build_model
from .train_clf import build_clf

BRANCHES = ["identity", "salt_expert", "blur_expert", "occ_expert"]


def load_parts(clf_ckpt, spec_dir, device):
    gck = torch.load(clf_ckpt, map_location=device)
    gate = build_clf(gck["cfg"]).to(device); gate.load_state_dict(gck["model"])
    experts, ecfgs = [], []
    for name in CLASSES[1:]:
        eck = torch.load(os.path.join(spec_dir, name, "best.pt"), map_location=device)
        m = build_model(eck["cfg"]).to(device); m.load_state_dict(eck["model"])
        experts.append(m); ecfgs.append(eck["cfg"])
    return gate, experts, gck["cfg"], ecfgs


def load_moe(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device)
    gate = build_clf(ck["gate_cfg"]); experts = [build_model(c) for c in ck["expert_cfgs"]]
    moe = SoftMoE(gate, experts, ck["cfg"]["temperature"]).to(device)
    moe.load_state_dict(ck["model"]); moe.eval()
    return moe, ck


def moe_loss(recon, y, w, logits, labels, cfg):
    l1 = F.l1_loss(recon, y)
    s = ssim(recon, y, data_range=1.0, size_average=True)
    ce = F.cross_entropy(logits / cfg["temperature"], labels)
    bal = ((w.mean(0) - 0.25) ** 2).sum()
    total = cfg["alpha"] * l1 + (1 - cfg["alpha"]) * (1 - s) + cfg["lambda_c"] * ce + cfg["lambda_b"] * bal
    return total, dict(l1=l1.item(), ssim=s.item(), ce=ce.item(), bal=bal.item())


@torch.no_grad()
def validate_moe(model, loader, device):
    model.eval(); n = 0; psnr = ssim_ = 0.0; wsum = torch.zeros(4, 4); cnt = torch.zeros(4); correct = 0
    for x, y, lab in loader:
        x, y, lab = x.to(device), y.to(device), lab.to(device)
        recon, w, _ = model(x)
        p, s = batch_metrics(recon, y); psnr += p.sum().item(); ssim_ += s.sum().item(); n += x.size(0)
        correct += (w.argmax(1) == lab).sum().item()
        for c in range(4):
            m = lab == c
            if m.any(): wsum[c] += w[m].sum(0).cpu(); cnt[c] += m.sum().item()
    W = (wsum / cnt.clamp_min(1)[:, None]).numpy()                      # rows: true class, cols: branch
    avg = (wsum.sum(0) / cnt.sum()).numpy()
    return dict(psnr=psnr / n, ssim=ssim_ / n, gate_acc=correct / n, W=W, avg_w=avg)


def run_training_moe(cfg, train_ds, val_ds, clf_ckpt, spec_dir, out_dir, device, run_name="moe",
                     experiment="task3_moe", trial=None, log_mlflow=True, num_workers=2, mlruns_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(cfg.get("seed", 42)); np.random.seed(cfg.get("seed", 42))
    gate, experts, gate_cfg, expert_cfgs = load_parts(clf_ckpt, spec_dir, device)
    model = SoftMoE(gate, experts, cfg["temperature"]).to(device)
    sampler = BalancedBatchSampler(len(train_ds) // 4, cfg["batch_size"], cfg.get("epoch_fraction", 0.25))
    train_dl = DataLoader(train_ds, batch_sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=num_workers)
    best_p, best, history, W_best = os.path.join(out_dir, "best.pt"), -1.0, [], None
    wu = cfg["warmup_epochs"]; opt = sched = None

    with mlflow_run(log_mlflow, experiment, run_name, mlruns_dir or out_dir):
        if log_mlflow:
            import mlflow
            mlflow.log_params(cfg)
        for ep in range(cfg["epochs"]):
            joint = ep >= wu
            if ep == 0 and not joint:                                   # stage 1: gate only
                model.freeze_experts = True
                for p in model.experts.parameters(): p.requires_grad_(False)
                opt = torch.optim.AdamW(model.gate.parameters(), lr=cfg["lr_warm"], weight_decay=cfg["weight_decay"])
            if ep == wu:                                                # stage 2: everything, smaller lr
                model.freeze_experts = False
                for p in model.experts.parameters(): p.requires_grad_(True)
                opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, cfg["epochs"] - wu))
            model.train()
            if not joint: model.experts.eval()                          # frozen experts keep fixed BN statistics
            t0 = time.time(); agg = dict(loss=0.0, l1=0.0, ssim=0.0, ce=0.0, bal=0.0); nb = 0
            for x, y, lab in train_dl:
                x, y, lab = x.to(device, non_blocking=True), y.to(device, non_blocking=True), lab.to(device)
                recon, w, logits = model(x)
                loss, parts = moe_loss(recon, y, w, logits, lab, cfg)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                agg["loss"] += loss.item(); nb += 1
                for k, v in parts.items(): agg[k] += v
            if sched is not None and joint: sched.step()
            v = validate_moe(model, val_dl, device)
            score = combined_score(v["psnr"], v["ssim"])
            collapsed = bool(v["avg_w"].max() > 0.6 or v["avg_w"].min() < 0.05)
            rec = {"epoch": ep, "stage": "joint" if joint else "warmup", **{f"train_{k}": a_ / nb for k, a_ in agg.items()},
                   "val_psnr": v["psnr"], "val_ssim": v["ssim"], "val_score": score, "gate_acc": v["gate_acc"],
                   "collapsed": int(collapsed), **{f"avg_w_{b}": float(v["avg_w"][i]) for i, b in enumerate(BRANCHES)}}
            history.append(rec)
            print(f"ep {ep:3d} [{rec['stage']:6s}] train {rec['train_loss']:.4f} | val PSNR {v['psnr']:.2f} SSIM {v['ssim']:.4f} "
                  f"score {score:.4f} | gate acc {v['gate_acc']:.3f} | avg w {np.round(v['avg_w'], 2)} | {time.time()-t0:.0f}s"
                  + ("  <-- ROUTING COLLAPSE" if collapsed else ""))
            if joint and score > best and not collapsed:
                best, W_best = score, v["W"]
                torch.save({"model": model.state_dict(), "cfg": cfg, "gate_cfg": gate_cfg, "expert_cfgs": expert_cfgs,
                            "epoch": ep, "best": best}, best_p)
            if log_mlflow:
                mlflow.log_metrics({k: float(x_) for k, x_ in rec.items() if k not in ("epoch", "stage")}, step=ep)
            if trial is not None and joint:
                import optuna
                trial.report(score, ep)
                if collapsed or trial.should_prune():
                    raise optuna.TrialPruned()
        if log_mlflow and os.path.exists(best_p):
            mlflow.log_metric("best_val_score", best); mlflow.log_artifact(best_p)
        json.dump(history, open(os.path.join(out_dir, "history.json"), "w"))
    return best, history
