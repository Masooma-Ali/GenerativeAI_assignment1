"""Evaluate the Task 2 hard-routed system in ORACLE and PREDICTED routing mode (optionally next to the
Task 1 universal model), per corruption type and severity, plus routing-failure analysis.

    python scripts/eval_task2.py --clf-ckpt clf_final/best.pt --spec-dir specialists \
        --universal-ckpt task1_spatial_final/best.pt --data-root data --split val --out task2_eval_val
    ... --split test --final --out task2_eval_test        # only once, for the final report"""
import argparse, os, sys
from collections import defaultdict
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.corruptions import CLASSES
from src.data.pets import PetRestorationDataset, load_manifest
from src.losses import batch_metrics
from src.pipeline import HardRouter
from src.train_ae import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--clf-ckpt", required=True); ap.add_argument("--spec-dir", required=True)
ap.add_argument("--universal-ckpt", default=None); ap.add_argument("--data-root", default="data")
ap.add_argument("--split", choices=["val", "test"], default="val"); ap.add_argument("--final", action="store_true")
ap.add_argument("--out", default="task2_eval")
a = ap.parse_args()
if a.split == "test" and not a.final:
    sys.exit("The test set must stay untouched until final evaluation. Re-run with --final when you are done tuning.")

device = "cuda" if torch.cuda.is_available() else "cpu"
proc = os.path.join(a.data_root, "processed")
images = np.load(os.path.join(proc, "trainval_128.npy" if a.split == "val" else "test_128.npy"))
manifest = load_manifest(os.path.join(proc, f"{a.split}_manifest.jsonl"))
ds = PetRestorationDataset(images, manifest=manifest)
router = HardRouter(a.clf_ckpt, a.spec_dir, device)
uni = None
if a.universal_ckpt:
    ck = torch.load(a.universal_ckpt, map_location=device)
    uni = build_model(ck["cfg"]).to(device); uni.load_state_dict(ck["model"]); uni.eval()

rec = defaultdict(list)
with torch.no_grad():
    for x, y, lab in DataLoader(ds, batch_size=256, shuffle=False, num_workers=2):
        x, y, lab = x.to(device), y.to(device), lab.to(device)
        out_p, probs, route = router(x, "predicted")
        out_o, _, _ = router(x, "oracle", lab)
        outs = [("in", x), ("oracle", out_o), ("pred", out_p)] + ([("uni", uni(x).clamp(0, 1))] if uni else [])
        for name, img in outs:
            p, s = batch_metrics(img, y); rec[f"psnr_{name}"] += p.cpu().tolist(); rec[f"ssim_{name}"] += s.cpu().tolist()
        rec["route_pred"] += route.cpu().tolist(); rec["true"] += lab.cpu().tolist()
        rec["p_max"] += probs.max(1).values.cpu().tolist()
df = pd.DataFrame(rec)
df["type"] = [e["cond"]["type"] for e in manifest]; df["severity"] = [e["cond"]["severity"] for e in manifest]
os.makedirs(a.out, exist_ok=True); df.to_csv(os.path.join(a.out, f"{a.split}_per_sample.csv"), index=False)

cols = [c for c in df.columns if c.startswith(("psnr_", "ssim_"))]
by_type = df.groupby("type")[cols].mean().round(3)
by_sev = df.groupby(["type", "severity"])[cols].mean().round(3)
corrupted = df[df.type != "clean"][cols].mean().round(3).to_frame("corrupted-only mean").T
by_type.to_csv(os.path.join(a.out, f"{a.split}_by_type.csv")); by_sev.to_csv(os.path.join(a.out, f"{a.split}_by_type_severity.csv"))
pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)
print("\n== by corruption type ==\n", by_type, "\n\n== by type and severity ==\n", by_sev, "\n\n", corrupted)
print("\n(clean PSNR is capped at 50 dB because a perfect copy has infinite PSNR)")

# ---- routing failures: predicted route != true corruption
fail = df[df.route_pred != df.true].copy()
fail["pred_class"] = [CLASSES[k] for k in fail.route_pred]
fail["psnr_loss_vs_oracle"] = fail.psnr_oracle - fail.psnr_pred
print(f"\nrouting errors: {len(fail)} / {len(df)} = {len(fail)/len(df):.4%}")
if len(fail):
    summ = fail.groupby(["type", "severity", "pred_class"]).agg(n=("true", "size"),
            mean_psnr_oracle=("psnr_oracle", "mean"), mean_psnr_pred=("psnr_pred", "mean"),
            mean_psnr_loss=("psnr_loss_vs_oracle", "mean"), mean_conf=("p_max", "mean")).round(3)
    print(summ); summ.to_csv(os.path.join(a.out, f"{a.split}_routing_failures.csv"))
    worst = fail.nlargest(6, "psnr_loss_vs_oracle").index.tolist()
    fig, axs = plt.subplots(len(worst), 4, figsize=(9, 2.3 * len(worst))); axs = np.atleast_2d(axs)
    for r, i in enumerate(worst):
        x, y, lab = ds[i]; xb = x[None].to(device)
        o, _, _ = router(xb, "oracle", torch.tensor([lab])); p, _, rt = router(xb, "predicted")
        row = df.loc[i]
        panels = [(y, "clean target"), (x, f"input: {row.type}/{row.severity}"),
                  (o[0].cpu(), f"oracle -> {CLASSES[lab]}  {row.psnr_oracle:.1f} dB"),
                  (p[0].cpu(), f"predicted -> {CLASSES[int(rt)]}  {row.psnr_pred:.1f} dB")]
        for c, (im, t) in enumerate(panels):
            axs[r, c].imshow(im.permute(1, 2, 0).numpy()); axs[r, c].set_title(t, fontsize=7); axs[r, c].axis("off")
    fig.suptitle("Routing failures with the largest quality loss", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(a.out, f"{a.split}_routing_failures.png"), dpi=140)
print("saved to", a.out)