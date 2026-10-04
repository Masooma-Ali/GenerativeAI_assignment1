"""Evaluate the Task 3 soft MoE: restoration quality (vs hard routing / universal AE), average gate weights per
corruption type and severity (heat-map), expert activity / dominance checks, example figures, and a
mixed-corruption stress test (where hard routing has to pick ONE expert).

    python scripts/eval_task3.py --moe-ckpt moe_final/best.pt --clf-ckpt clf_final/best.pt --spec-dir specialists \
        --universal-ckpt task1_spatial_final/best.pt --data-root data --split val --out task3_eval_val
    ... --split test --final --out task3_eval_test      # only once, for the final report"""
import argparse, os, sys
from collections import defaultdict
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.corruptions import CLASSES, apply_condition, sample_boxes
from src.data.pets import PetRestorationDataset, load_manifest
from src.losses import batch_metrics
from src.pipeline import HardRouter
from src.train_ae import build_model
from src.train_moe import BRANCHES, load_moe

ap = argparse.ArgumentParser()
ap.add_argument("--moe-ckpt", required=True); ap.add_argument("--clf-ckpt", required=True)
ap.add_argument("--spec-dir", required=True); ap.add_argument("--universal-ckpt", default=None)
ap.add_argument("--data-root", default="data"); ap.add_argument("--split", choices=["val", "test"], default="val")
ap.add_argument("--final", action="store_true"); ap.add_argument("--out", default="task3_eval")
ap.add_argument("--mixed-n", type=int, default=300)
a = ap.parse_args()
if a.split == "test" and not a.final:
    sys.exit("The test set must stay untouched until final evaluation. Re-run with --final when you are done tuning.")

device = "cuda" if torch.cuda.is_available() else "cpu"
proc = os.path.join(a.data_root, "processed")
images = np.load(os.path.join(proc, "trainval_128.npy" if a.split == "val" else "test_128.npy"))
manifest = load_manifest(os.path.join(proc, f"{a.split}_manifest.jsonl"))
ds = PetRestorationDataset(images, manifest=manifest)
moe, _ = load_moe(a.moe_ckpt, device)
hard = HardRouter(a.clf_ckpt, a.spec_dir, device)
uni = None
if a.universal_ckpt:
    ck = torch.load(a.universal_ckpt, map_location=device)
    uni = build_model(ck["cfg"]).to(device); uni.load_state_dict(ck["model"]); uni.eval()
os.makedirs(a.out, exist_ok=True)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", None)


def run_models(x):
    """-> dict name -> restored batch, plus soft weights and hard route."""
    soft, w, _ = moe(x); hp, _, route = hard(x, "predicted")
    outs = {"in": x, "soft": soft.clamp(0, 1), "hard": hp}
    if uni is not None: outs["uni"] = uni(x).clamp(0, 1)
    return outs, w, route


# ---------------------------------------------------------------- main evaluation
rec = defaultdict(list)
with torch.no_grad():
    for x, y, lab in DataLoader(ds, batch_size=128, shuffle=False, num_workers=2):
        x, y = x.to(device), y.to(device)
        outs, w, route = run_models(x)
        for name, img in outs.items():
            p, s = batch_metrics(img, y); rec[f"psnr_{name}"] += p.cpu().tolist(); rec[f"ssim_{name}"] += s.cpu().tolist()
        for i, b in enumerate(BRANCHES): rec[f"w_{b}"] += w[:, i].cpu().tolist()
        rec["true"] += lab.tolist(); rec["hard_route"] += route.cpu().tolist()
df = pd.DataFrame(rec)
df["type"] = [e["cond"]["type"] for e in manifest]; df["severity"] = [e["cond"]["severity"] for e in manifest]
df.to_csv(os.path.join(a.out, f"{a.split}_per_sample.csv"), index=False)
mcols = [c for c in df.columns if c.startswith(("psnr_", "ssim_"))]; wcols = [f"w_{b}" for b in BRANCHES]
by_type = df.groupby("type")[mcols].mean().round(3); by_sev = df.groupby(["type", "severity"])[mcols].mean().round(3)
by_type.to_csv(os.path.join(a.out, f"{a.split}_by_type.csv")); by_sev.to_csv(os.path.join(a.out, f"{a.split}_by_type_severity.csv"))
print("\n== quality by corruption type ==\n", by_type, "\n\n== quality by type and severity ==\n", by_sev)
print("\ncorrupted-only mean:\n", df[df.type != "clean"][mcols].mean().round(3).to_frame("mean").T)
print("(clean PSNR is capped at 50 dB)")

# ---------------------------------------------------------------- gate behaviour
W = df.groupby(["type", "severity"])[wcols].mean().round(3); W.to_csv(os.path.join(a.out, f"{a.split}_gate_weights.csv"))
print("\n== average gate weights per true corruption type and severity ==\n", W)
fig, ax = plt.subplots(figsize=(6, 0.45 * len(W) + 1.5)); im = ax.imshow(W.values, vmin=0, vmax=1, cmap="viridis")
ax.set_xticks(range(4)); ax.set_xticklabels(BRANCHES, rotation=25); ax.set_yticks(range(len(W)))
ax.set_yticklabels([f"{t}/{s}" for t, s in W.index], fontsize=8)
for i in range(W.shape[0]):
    for j in range(4): ax.text(j, i, f"{W.values[i, j]:.2f}", ha="center", va="center", color="w" if W.values[i, j] < .6 else "k", fontsize=8)
fig.colorbar(im); ax.set_title("Mean routing weight"); fig.tight_layout()
fig.savefig(os.path.join(a.out, f"{a.split}_routing_heatmap.png"), dpi=150); plt.close(fig)

wm = df[wcols].values; amax = wm.argmax(1)
act = pd.DataFrame({"mean_weight": wm.mean(0).round(3), "share_argmax": np.bincount(amax, minlength=4) / len(df),
                    "mean_weight_when_not_its_class": [wm[df.true.values != k, k].mean() for k in range(4)]}, index=BRANCHES).round(3)
print("\n== expert activity / dominance ==\n", act); act.to_csv(os.path.join(a.out, f"{a.split}_expert_activity.csv"))
for b, row in act.iterrows():
    if row.mean_weight < 0.05: print(f"WARNING: {b} looks INACTIVE (mean weight {row.mean_weight})")
    if row.mean_weight > 0.5: print(f"WARNING: {b} DOMINATES (mean weight {row.mean_weight})")
conf = pd.crosstab(pd.Series(df.true.map(dict(enumerate(CLASSES))), name="true"),
                   pd.Series([BRANCHES[k] for k in amax], name="top branch"), normalize="index").round(3)
print("\n== which branch has the largest weight, per true class ==\n", conf); conf.to_csv(os.path.join(a.out, f"{a.split}_top_branch.csv"))

# ---------------------------------------------------------------- example figures
ent = -(wm * np.log(wm + 1e-9)).sum(1); nonclean = (df.type != "clean").values
dom = np.where(nonclean & (wm.max(1) > 0.95))[0]; dist = np.where(nonclean)[0][np.argsort(-ent[nonclean])[:4]]
rng = np.random.default_rng(0); pick = list(rng.choice(dom, min(4, len(dom)), replace=False)) if len(dom) else []
pick = [("one expert dominates", i) for i in pick] + [("weights distributed", i) for i in dist]
fig, axs = plt.subplots(len(pick), 5, figsize=(11, 2.4 * len(pick))); axs = np.atleast_2d(axs)
for r, (kind, i) in enumerate(pick):
    x, y, _ = ds[int(i)]
    with torch.no_grad():
        outs, w, _ = run_models(x[None].to(device))
    ws = " ".join(f"{b[:3]} {v:.2f}" for b, v in zip(BRANCHES, w[0].cpu().tolist()))
    panels = [(y, "clean target"), (x, f"{df.type[i]}/{df.severity[i]}"), (outs["soft"][0].cpu(), f"soft {df.psnr_soft[i]:.1f} dB"),
              (outs["hard"][0].cpu(), f"hard {df.psnr_hard[i]:.1f} dB")]
    for c, (im_, t) in enumerate(panels):
        axs[r, c].imshow(im_.permute(1, 2, 0).numpy()); axs[r, c].set_title(t, fontsize=7); axs[r, c].axis("off")
    axs[r, 4].bar(range(4), w[0].cpu().numpy(), color=["gray", "tab:red", "tab:blue", "tab:green"]); axs[r, 4].set_ylim(0, 1)
    axs[r, 4].set_xticks(range(4)); axs[r, 4].set_xticklabels(["id", "salt", "blur", "occ"], fontsize=7)
    axs[r, 4].set_title(kind, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(a.out, f"{a.split}_gate_examples.png"), dpi=140); plt.close(fig)

# ---------------------------------------------------------------- mixed-corruption stress test
ids = list(dict.fromkeys(e["img_idx"] for e in manifest))[:a.mixed_n]
def mixed(img, kind, seed):
    r = np.random.default_rng(seed); blur = {"type": "blur", "kernel": 5, "sigma": 1.5, "seed": seed}
    salt = {"type": "salt_pepper", "p": 0.08, "seed": seed}; occ = {"type": "occlusion", "boxes": sample_boxes(r, 2, 0.2), "seed": seed}
    seq = {"blur+salt_pepper": [blur, salt], "salt_pepper+occlusion": [salt, occ], "blur+occlusion": [blur, occ]}[kind]
    for c in seq: img = apply_condition(img, c)
    return img
to_t = lambda im: torch.from_numpy(im).permute(2, 0, 1).float().div(255)
rows = []
with torch.no_grad():
    for kind in ["blur+salt_pepper", "salt_pepper+occlusion", "blur+occlusion"]:
        r_ = defaultdict(list)
        for k, idx in enumerate(ids):
            y = to_t(images[idx])[None].to(device); x = to_t(mixed(images[idx], kind, 1000 + k))[None].to(device)
            outs, w, route = run_models(x)
            for name, img in outs.items():
                p, s = batch_metrics(img, y); r_[f"psnr_{name}"].append(p.item()); r_[f"ssim_{name}"].append(s.item())
            r_["hard_picks"].append(CLASSES[int(route)]); 
            for i_, b in enumerate(BRANCHES): r_[f"w_{b}"].append(w[0, i_].item())
        row = {"mix": kind, **{k: np.mean(v) for k, v in r_.items() if k.startswith(("psnr", "ssim", "w_"))},
               "hard_most_common": pd.Series(r_["hard_picks"]).mode()[0]}
        rows.append(row)
mix = pd.DataFrame(rows).set_index("mix").round(3); mix.to_csv(os.path.join(a.out, f"{a.split}_mixed_corruption.csv"))
print(f"\n== mixed-corruption stress test ({len(ids)} images each) ==\n", mix.T)
print("saved to", a.out)
