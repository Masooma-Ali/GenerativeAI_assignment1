"""Training loop for the Task 4 conditional GAN (BCE-with-logits adversarial loss + lambda_l1 * L1)."""
import json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from .models.cgan import UNetGenerator, build_gan
from .tracking import mlflow_run


def to01(x):
    return ((x + 1) / 2).clamp(0, 1)


def metrics01(fake, real):
    """fake/real in [-1,1], 1 channel. Returns per-image (L1, SSIM, PSNR) on the [0,1] scale."""
    f, r = to01(fake), to01(real)
    l1 = (f - r).abs().flatten(1).mean(1)
    mse = ((f - r) ** 2).flatten(1).mean(1).clamp_min(1e-8)
    return l1, ssim(f, r, data_range=1.0, size_average=False), -10 * torch.log10(mse)


@torch.no_grad()
def validate(G, D, loader, device, bce):
    G.eval(); D.eval(); n = 0; tot = dict(l1=0.0, ssim=0.0, psnr=0.0, d_real=0.0, d_fake=0.0)
    for x, y, s in loader:
        x, y, s = x.to(device), y.to(device), s.to(device)
        fake = G(x, s); l1, ss, ps = metrics01(fake, y); b = x.size(0); n += b
        pr, pf = D(x, y, s), D(x, fake, s)
        tot["l1"] += l1.sum().item(); tot["ssim"] += ss.sum().item(); tot["psnr"] += ps.sum().item()
        tot["d_real"] += bce(pr, torch.ones_like(pr)).item() * b; tot["d_fake"] += bce(pf, torch.zeros_like(pf)).item() * b
    return {k: v / n for k, v in tot.items()}


@torch.no_grad()
def sample_grid(G, val_ds, device, n=8):
    """Same validation photos every time: rows = photo, ground-truth sketch, generated style 0/1/2."""
    G.eval()
    x = torch.stack([val_ds[i][0] for i in range(n)]).to(device); y = torch.stack([val_ds[i][1] for i in range(n)])
    rows = [to01(x.cpu()), to01(y).repeat(1, 3, 1, 1)]
    for st in range(3):
        rows.append(to01(G(x, torch.full((n,), st, device=device, dtype=torch.long)).cpu()).repeat(1, 3, 1, 1))
    g = make_grid(torch.cat(rows), nrow=n, padding=2)
    return (g.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def run_training_gan(cfg, train_ds, val_ds, out_dir, device, run_name="cgan", experiment="task4_cgan", trial=None,
                     log_mlflow=True, num_workers=2, mlruns_dir=None, resume=False, sample_every=5):
    os.makedirs(os.path.join(out_dir, "samples"), exist_ok=True)
    torch.manual_seed(cfg.get("seed", 42)); np.random.seed(cfg.get("seed", 42))
    G, D = build_gan(cfg, device)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg["lr_g"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=(0.5, 0.999))
    bce, l1_fn = nn.BCEWithLogitsLoss(), nn.L1Loss()
    train_dl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                          num_workers=num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=num_workers)
    last_p, best_p = os.path.join(out_dir, "last.pt"), os.path.join(out_dir, "best.pt")
    start, best, history = 0, -1e9, []
    if resume and os.path.exists(last_p):
        ck = torch.load(last_p, map_location=device)
        G.load_state_dict(ck["G"]); D.load_state_dict(ck["D"]); opt_g.load_state_dict(ck["opt_g"]); opt_d.load_state_dict(ck["opt_d"])
        start, best = ck["epoch"] + 1, ck["best"]; print("resumed from epoch", start)

    with mlflow_run(log_mlflow, experiment, run_name, mlruns_dir or out_dir):
        if log_mlflow:
            import mlflow
            mlflow.log_params(cfg)
            mlflow.log_param("n_params_G", sum(p.numel() for p in G.parameters()))
            mlflow.log_param("n_params_D", sum(p.numel() for p in D.parameters()))
        for ep in range(start, cfg["epochs"]):
            G.train(); D.train(); t0 = time.time(); agg = dict(d_real=0.0, d_fake=0.0, g_adv=0.0, g_l1=0.0); nb = 0
            for x, y, s in train_dl:
                x, y, s = x.to(device, non_blocking=True), y.to(device, non_blocking=True), s.to(device)
                fake = G(x, s)
                pr, pf = D(x, y, s), D(x, fake.detach(), s)                       # --- discriminator step
                d_real, d_fake = bce(pr, torch.ones_like(pr)), bce(pf, torch.zeros_like(pf))
                opt_d.zero_grad(set_to_none=True); (0.5 * (d_real + d_fake)).backward(); opt_d.step()
                pg = D(x, fake, s)                                                  # --- generator step
                g_adv, g_l1 = bce(pg, torch.ones_like(pg)), l1_fn(fake, y)
                opt_g.zero_grad(set_to_none=True); (g_adv + cfg["lambda_l1"] * g_l1).backward(); opt_g.step()
                for k, v in zip(agg, (d_real, d_fake, g_adv, g_l1)): agg[k] += v.item()
                nb += 1
            v = validate(G, D, val_dl, device, bce)
            score = v["ssim"] - v["l1"]                                             # selection / Optuna objective
            rec = {"epoch": ep, **{f"train_{k}": a / nb for k, a in agg.items()},
                   **{f"val_{k}": x_ for k, x_ in v.items()}, "val_score": score}
            history.append(rec)
            print(f"ep {ep:3d} | D real {rec['train_d_real']:.3f} fake {rec['train_d_fake']:.3f} | G adv {rec['train_g_adv']:.3f} "
                  f"L1 {rec['train_g_l1']:.4f} | val L1 {v['l1']:.4f} SSIM {v['ssim']:.4f} PSNR {v['psnr']:.2f} score {score:.4f} | {time.time()-t0:.0f}s")
            if score > best:
                best = score
                torch.save({"G": G.state_dict(), "cfg": cfg, "epoch": ep, "best": best}, best_p)
            torch.save({"G": G.state_dict(), "D": D.state_dict(), "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(),
                        "cfg": cfg, "epoch": ep, "best": best}, last_p)
            if ep % sample_every == 0 or ep == cfg["epochs"] - 1:
                img = sample_grid(G, val_ds, device)
                from PIL import Image
                Image.fromarray(img).save(os.path.join(out_dir, "samples", f"epoch_{ep:03d}.png"))
                if log_mlflow: mlflow.log_image(img, f"samples/epoch_{ep:03d}.png")
            if log_mlflow: mlflow.log_metrics({k: float(x_) for k, x_ in rec.items() if k != "epoch"}, step=ep)
            if trial is not None:
                trial.report(score, ep)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()
        if log_mlflow:
            mlflow.log_metric("best_val_score", best)
            if os.path.exists(best_p): mlflow.log_artifact(best_p)
        json.dump(history, open(os.path.join(out_dir, "history.json"), "w"))
    return best, history


def load_generator(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device); cfg = ck["cfg"]
    G = UNetGenerator(cfg["base"], cfg["style_dim"], cfg["dropout"]).to(device)
    G.load_state_dict(ck["G"]); G.eval()
    return G, ck
