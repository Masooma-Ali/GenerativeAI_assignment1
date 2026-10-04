"""Generate the report figures that are not produced by the evaluation scripts:
training curves, the Task 2 confusion matrix (from the per-sample test CSV) and the
architecture / system diagrams.

    python scripts/make_report_figures.py --root <folder with the unzipped task*_all folders> --out report/figures

Expected inputs under --root (the Kaggle output zips):
    task1_all/task1_spatial_final/history.json, task1_all/optuna_task1/trials/trial_020/history.json
    task2_complete/task2_eval_test/test_per_sample.csv
    task3_all/moe_final/history.json
    task4_all/gan_final/history.json
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 220, "savefig.bbox": "tight"})
C = {"blue": "#2563eb", "orange": "#ea580c", "green": "#16a34a", "red": "#dc2626", "purple": "#9333ea",
     "grey": "#64748b", "teal": "#0d9488"}
CLASSES = ["clean", "salt-pepper", "blur", "occlusion"]


def load(path):
    with open(path) as f:
        return json.load(f)


def col(h, k):
    return np.array([e[k] for e in h])


# ------------------------------------------------------------------------------------ curves
def t1_curves(root, out):
    sp = load(os.path.join(root, "task1_all/task1_spatial_final/history.json"))
    fl = load(os.path.join(root, "task1_all/optuna_task1/trials/trial_020/history.json"))
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
    ax[0].plot(col(sp, "epoch") + 1, col(sp, "val_score"), color=C["blue"], label="spatial 8x8x64 (final)")
    ax[0].plot(col(fl, "epoch") + 1, col(fl, "val_score"), color=C["orange"], label="flat 2048-d (best trial)")
    ax[0].set(title="validation score S", xlabel="epoch")
    ax[1].plot(col(sp, "epoch") + 1, col(sp, "val_psnr"), color=C["blue"])
    ax[1].plot(col(fl, "epoch") + 1, col(fl, "val_psnr"), color=C["orange"])
    ax[1].set(title="validation PSNR (dB)", xlabel="epoch")
    ax[2].plot(col(sp, "epoch") + 1, col(sp, "train_loss"), color=C["blue"], label="train")
    ax[2].plot(col(sp, "epoch") + 1, col(sp, "val_loss"), color=C["blue"], ls="--", label="val")
    ax[2].set(title="loss (spatial model)", xlabel="epoch")
    ax[0].legend(fontsize=7, frameon=False); ax[2].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "t1_curves.png")); plt.close(fig)


def t2_confusion(root, out):
    d = pd.read_csv(os.path.join(root, "task2_complete/task2_eval_test/test_per_sample.csv"))
    cm = pd.crosstab(d["true"], d["route_pred"]).reindex(index=range(4), columns=range(4), fill_value=0).values
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    ax.grid(False)
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cmn[i, j]:.4f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xticks(range(4), CLASSES, rotation=30, ha="right"); ax.set_yticks(range(4), CLASSES)
    ax.set(xlabel="predicted", ylabel="true", title=f"Test set, accuracy {np.trace(cm) / cm.sum():.4f}")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(os.path.join(out, "t2_confusion_test.png")); plt.close(fig)


def t3_curves(root, out):
    h = load(os.path.join(root, "task3_all/moe_final/history.json"))
    ep = col(h, "epoch") + 1
    n_warm = sum(e["stage"] == "warmup" for e in h)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
    for a in ax:
        a.axvspan(0.5, n_warm + 0.5, color=C["grey"], alpha=0.15, lw=0)
    ax[0].plot(ep, col(h, "val_score"), color=C["blue"]); ax[0].set(title="validation score S", xlabel="epoch")
    ax[1].plot(ep, col(h, "train_loss"), color=C["blue"], label="total")
    ax[1].plot(ep, col(h, "train_ce"), color=C["orange"], label="CE")
    ax[1].plot(ep, col(h, "train_bal") * 10, color=C["green"], label="balance x10")
    ax[1].set(title="training loss terms", xlabel="epoch"); ax[1].legend(fontsize=7, frameon=False)
    for k, lab, c in [("avg_w_identity", "identity", C["grey"]), ("avg_w_salt_expert", "salt", C["red"]),
                      ("avg_w_blur_expert", "blur", C["blue"]), ("avg_w_occ_expert", "occlusion", C["green"])]:
        ax[2].plot(ep, col(h, k), color=c, label=lab)
    ax[2].axhline(0.25, color="k", lw=0.6, ls=":")
    ax[2].set(title="mean validation gate weight", xlabel="epoch", ylim=(0.15, 0.35))
    ax[2].legend(fontsize=6.5, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out, "t3_curves.png")); plt.close(fig)


def t4_curves(root, out):
    h = load(os.path.join(root, "task4_all/gan_final/history.json"))
    ep = col(h, "epoch") + 1
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.2))
    ax[0].plot(ep, col(h, "train_d_real"), color=C["blue"], label="D real")
    ax[0].plot(ep, col(h, "train_d_fake"), color=C["orange"], label="D fake")
    ax[0].plot(ep, col(h, "train_g_adv"), color=C["red"], label="G adversarial")
    ax[0].set(title="adversarial losses (train)", xlabel="epoch"); ax[0].legend(fontsize=7, frameon=False)
    ax[1].plot(ep, col(h, "train_g_l1"), color=C["purple"], label="G L1 (train)")
    ax[1].plot(ep, col(h, "val_l1"), color=C["purple"], ls="--", label="L1 (val)")
    ax[1].set(title="reconstruction L1", xlabel="epoch"); ax[1].legend(fontsize=7, frameon=False)
    ax[2].plot(ep, col(h, "val_ssim"), color=C["teal"], label="SSIM")
    ax[2].plot(ep, col(h, "val_score"), color=C["blue"], label="score SSIM-L1")
    ax[2].set(title="validation metrics", xlabel="epoch"); ax[2].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "t4_curves.png")); plt.close(fig)


# ---------------------------------------------------------------------------------- diagrams
def box(ax, x, y, w, h, text, fc, ec="#334155", fs=7, tc="black", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc,
            weight="bold" if bold else "normal")


def arrow(ax, x0, y0, x1, y1, color="#334155", style="-|>", ls="-", lw=0.9, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=8, color=color,
                                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}"))


def t1_arch(out):
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 3.9))
    specs = [
        ("(a) Flat baseline: fully connected bottleneck (latent 256-4096)",
         [(128, 3), (64, 32), (32, 64), (16, 128), (8, 256), (4, 256)], "flatten + Linear\nz ∈ R^256..4096"),
        ("(b) Final: spatial bottleneck 8×8×64 = 4 096 values (12× compression)",
         [(128, 3), (64, 48), (32, 96), (16, 192), (8, 384)], "1×1 conv\n8×8×64"),
    ]
    for ax, (title, enc, neck) in zip(axs, specs):
        ax.set_xlim(-0.2, 21.5); ax.set_ylim(-1.1, 2.9); ax.axis("off")
        ax.text(-0.1, 2.85, title, fontsize=8, weight="bold", va="top")
        layers = [(s, c, "#e2e8f0" if i == 0 else "#dbeafe") for i, (s, c) in enumerate(enc)]
        dec = [(s, c, "#dcfce7") for s, c in reversed(enc[:-1])] + []
        dec[-1] = (128, 3, "#e2e8f0")
        x, y = 0.0, 0.9
        def slab(x, s, c, fc, k):
            h = 0.35 + s / 128 * 1.75; w = 0.25 + min(c, 384) / 384 * 0.35
            ax.add_patch(Rectangle((x, y - h / 2), w, h, fc=fc, ec="#334155", lw=0.6))
            lab = f"{s}²×{c}"
            ax.text(x + w / 2, y - h / 2 - 0.08 - 0.32 * (k % 2), lab, ha="center", va="top", fontsize=5.5)
            return x + w
        for k, (s, c, fc) in enumerate(layers):
            x = slab(x, s, c, fc, k); arrow(ax, x + 0.05, y, x + 0.55, y); x += 0.6
        box(ax, x, y - 0.5, 2.6, 1.0, neck, "#fde68a", fs=6.5, bold=True); x += 2.6
        x_dec = x + 0.6
        arrow(ax, x + 0.05, y, x + 0.55, y); x += 0.6
        for k, (s, c, fc) in enumerate(dec):
            x = slab(x, s, c, fc, k)
            if k < len(dec) - 1:
                arrow(ax, x + 0.05, y, x + 0.55, y); x += 0.6
        ax.text(x + 0.15, y, "sigmoid\noutput", fontsize=5.5, va="center")
        ax.text(0.2, 2.25, "encoder: stride-2 conv + conv (BN, LeakyReLU)", fontsize=6, color=C["blue"])
        ax.text(x_dec, 2.25, "decoder: NN-upsample + 2 conv (BN, LeakyReLU)", fontsize=6, color=C["green"])
        ax.set_xlim(-0.2, max(x + 1.6, x_dec + 7.5))
    axs[1].text(10.6, -1.05, "no skip connections in either model: all information passes through the bottleneck",
                fontsize=6, ha="center", style="italic", color=C["grey"])
    fig.tight_layout(); fig.savefig(os.path.join(out, "t1_arch.png")); plt.close(fig)


def t4_arch(out):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 22); ax.set_ylim(0, 11.2); ax.axis("off")
    ax.text(0.2, 11.1, "Generator G(x, s): U-Net, 7 down / 6 up blocks", fontsize=8, weight="bold", va="top")
    box(ax, 0.2, 6.9, 2.5, 1.1, "photo x\n3×128×128", "#e2e8f0", fs=6.5)
    box(ax, 0.2, 5.1, 2.5, 1.1, "style s ∈ {0,1,2}\nEmbedding(3, d)", "#fde68a", fs=6.3)
    box(ax, 3.2, 6.0, 1.4, 1.2, "tile +\nconcat", "#fef3c7", fs=6.3)
    arrow(ax, 2.7, 7.4, 3.2, 6.9); arrow(ax, 2.7, 5.7, 3.2, 6.3)
    top = 8.9
    enc = [64, 32, 16, 8, 4, 2, 1]; dec = [2, 4, 8, 16, 32, 64]
    ex, dx = [], []
    for i, s in enumerate(enc):
        x = 5.1 + i * 0.72; h = 0.5 + s / 64 * 3.0
        ax.add_patch(Rectangle((x, top - h), 0.45, h, fc="#dbeafe", ec="#334155", lw=0.6)); ex.append(x + 0.225)
    for i, s in enumerate(dec):
        x = 10.4 + i * 0.72; h = 0.5 + s / 64 * 3.0
        ax.add_patch(Rectangle((x, top - h), 0.45, h, fc="#dcfce7", ec="#334155", lw=0.6)); dx.append(x + 0.225)
    for i in range(6):                          # skip: encoder level -> mirrored decoder level
        arrow(ax, ex[5 - i], top + 0.05, dx[i], top + 0.05, color=C["grey"], lw=0.6, rad=-0.35)
    arrow(ax, 4.6, 6.6, 5.1, 7.2)
    ax.text(7.4, 5.0, "encoder: conv 4×4/2,\nBN, LeakyReLU", fontsize=5.8, ha="center", color=C["blue"])
    ax.text(12.4, 5.0, "decoder: ConvT + FiLM-BN + ReLU,\ndropout in first 3 blocks", fontsize=5.8, ha="center", color=C["green"])
    ax.text(9.9, 10.55, "skip connections", fontsize=5.8, ha="center", color=C["grey"])
    box(ax, 15.0, 7.0, 2.4, 1.2, "ConvT → tanh\nsketch ŷ: 1×128²", "#e2e8f0", fs=6.2)
    arrow(ax, 14.75, 7.6, 15.0, 7.6)
    arrow(ax, 2.7, 5.3, 12.4, 4.35, color=C["orange"], rad=0.08)
    ax.text(6.6, 4.05, "FiLM: γ(e), β(e) modulate every decoder block", fontsize=6, color=C["orange"])
    box(ax, 17.7, 5.4, 4.2, 3.2, "Selected (Optuna #9)\nlr_G 5.8e-4, lr_D 6.6e-4\nbatch 16, base b = 64\n"
                                 "dropout 0.47, style dim d = 32\n150 epochs, Adam β=(0.5, 0.999)", "#f8fafc", fs=6.0)
    ax.text(0.2, 3.1, "Discriminator D(x, y, s): conditional PatchGAN", fontsize=8, weight="bold", va="top")
    box(ax, 0.1, 0.4, 3.45, 1.7, "concat: photo (3) +\nreal y or generated ŷ (1)\n+ tiled own embedding (d)", "#fef3c7", fs=5.8)
    prev = 3.5
    for i, lab in enumerate(["64²×b", "32²×2b", "16²×4b", "15²×8b"]):
        x0 = 4.0 + i * 1.85
        arrow(ax, prev, 1.25, x0, 1.25)
        box(ax, x0, 0.65, 1.5, 1.2, f"conv 4×4\n{lab}", "#fee2e2", fs=5.8); prev = x0 + 1.5
    arrow(ax, prev, 1.25, 11.8, 1.25)
    box(ax, 11.8, 0.65, 2.4, 1.2, "14×14 patch\nlogits (BCE)", "#fecaca", fs=6.3, bold=True)
    box(ax, 15.0, 0.35, 6.8, 2.2, "L_D = ½[BCE(D(x,s,y),1) + BCE(D(x,s,ŷ),0)]\n"
                                  "L_G = BCE(D(x,s,ŷ),1) + λ_L1 ‖y − ŷ‖₁\n"
                                  "final λ_L1 = 272.5", "#f8fafc", fs=6.2)
    fig.tight_layout(); fig.savefig(os.path.join(out, "t4_arch.png")); plt.close(fig)


def app_architecture(out):
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.set_xlim(0, 24); ax.set_ylim(0, 10); ax.axis("off")
    box(ax, 0.2, 3.6, 3.4, 2.8, "Browser\n\nReact + Tailwind\nsingle-page app\n(4 workspaces +\nHealth page)", "#e0e7ff", fs=6.5)
    ax.add_patch(FancyBboxPatch((4.4, 0.4), 19.3, 9.0, boxstyle="round,pad=0.02,rounding_size=0.2",
                                fc="none", ec=C["blue"], lw=1.0, ls="--"))
    ax.text(4.7, 9.0, "docker compose up --build", fontsize=7, color=C["blue"], weight="bold")
    box(ax, 5.0, 3.0, 4.6, 4.0, "frontend container\nnginx:1.27\n\n• static Vite build\n• /api/* → backend:8000\n• port 8080", "#dbeafe", fs=6.3)
    box(ax, 11.0, 3.0, 5.4, 4.0, "backend container\npython 3.11 + FastAPI\n\n• validate upload, resize 128²\n"
                                  "• runtime corruption\n  (src/data/corruptions.py)\n• PSNR/SSIM, timing\n• port 8000 (/docs)", "#dcfce7", fs=6.0)
    box(ax, 17.8, 5.0, 5.6, 3.6, "ONNX Runtime (CPU)\n\ntask1_universal\ntask2_classifier + 3 specialists\n"
                                  "task3_soft_moe\ntask4_face2sketch", "#fef3c7", fs=6.0)
    box(ax, 17.8, 1.0, 5.6, 2.8, "host volumes (read-only)\n./src/models → /app/models\n./backend/samples", "#f1f5f9", fs=6.0)
    arrow(ax, 3.6, 5.0, 5.0, 5.0, style="<|-|>"); ax.text(4.3, 5.3, "HTTP", fontsize=6, ha="center")
    arrow(ax, 9.6, 5.0, 11.0, 5.0, style="<|-|>"); ax.text(10.3, 5.3, "proxy", fontsize=6, ha="center")
    arrow(ax, 16.4, 5.8, 17.8, 6.6, style="<|-|>")
    arrow(ax, 20.6, 3.8, 20.6, 5.0)
    arrow(ax, 13.7, 3.0, 17.8, 2.2, style="<|-", rad=-0.2)
    ax.text(9.5, 1.3, "backend healthcheck: GET /api/health\nfrontend starts after backend is healthy",
            fontsize=5.8, ha="center", color=C["grey"])
    fig.tight_layout(); fig.savefig(os.path.join(out, "app_architecture.png")); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder containing the unzipped task*_all folders")
    ap.add_argument("--out", default="report/figures")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for fn in (t1_curves, t2_confusion, t3_curves, t4_curves):
        fn(a.root, a.out)
    for fn in (t1_arch, t4_arch, app_architecture):
        fn(a.out)
    print("figures written to", a.out)
