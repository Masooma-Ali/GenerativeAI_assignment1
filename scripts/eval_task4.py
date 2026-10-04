"""Evaluate the Task 4 generator: L1 / SSIM / PSNR overall and per style, a trivial baseline (grayscale photo),
a STYLE-CONDITIONING check (every photo generated with all three styles and compared with the ground truth),
example grids and the worst failure cases.

    python scripts/eval_task4.py --ckpt gan_final/best.pt --fs2k fs2k --split val --out task4_eval_val
    python scripts/eval_task4.py --ckpt gan_final/best.pt --fs2k fs2k --split test --final --out task4_eval_test"""
import argparse, os, sys
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.fs2k import FS2KDataset
from src.train_gan import load_generator, metrics01, to01

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--fs2k", required=True)
ap.add_argument("--split", choices=["val", "test"], default="val"); ap.add_argument("--final", action="store_true")
ap.add_argument("--out", default="task4_eval")
a = ap.parse_args()
if a.split == "test" and not a.final:
    sys.exit("The official test set must stay untouched until final evaluation. Re-run with --final when you are done tuning.")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", None)
device = "cuda" if torch.cuda.is_available() else "cpu"
ds = FS2KDataset(os.path.join(a.fs2k, f"{a.split}.npz")); G, ck = load_generator(a.ckpt, device)
os.makedirs(a.out, exist_ok=True)
print("style counts in", a.split, np.bincount(ds.styles, minlength=3))

rec = {k: [] for k in ["style", "l1", "ssim", "psnr", "base_l1", "base_ssim"] +
       [f"{m}_c{c}" for m in ("l1", "ssim") for c in range(3)] + ["diff_01", "diff_02", "diff_12"]}
with torch.no_grad():
    for x, y, s in DataLoader(ds, batch_size=64, shuffle=False, num_workers=2):
        x, y, s = x.to(device), y.to(device), s.to(device)
        outs = [G(x, torch.full_like(s, c)) for c in range(3)]
        own = torch.stack(outs, 1)[torch.arange(len(s)), s]                          # generated with the TRUE style
        l1, ss, ps = metrics01(own, y)
        gray = (0.299 * to01(x)[:, 0] + 0.587 * to01(x)[:, 1] + 0.114 * to01(x)[:, 2])[:, None]
        bl1, bss, _ = metrics01(gray * 2 - 1, y)
        for k, v in [("style", s), ("l1", l1), ("ssim", ss), ("psnr", ps), ("base_l1", bl1), ("base_ssim", bss)]:
            rec[k] += v.cpu().tolist()
        for c in range(3):
            l1c, ssc, _ = metrics01(outs[c], y); rec[f"l1_c{c}"] += l1c.cpu().tolist(); rec[f"ssim_c{c}"] += ssc.cpu().tolist()
        for (i, j) in [(0, 1), (0, 2), (1, 2)]:
            rec[f"diff_{i}{j}"] += (to01(outs[i]) - to01(outs[j])).abs().flatten(1).mean(1).cpu().tolist()
df = pd.DataFrame(rec); df.to_csv(os.path.join(a.out, f"{a.split}_per_sample.csv"), index=False)

main = df[["l1", "ssim", "psnr", "base_l1", "base_ssim"]].mean().round(4)
by_style = df.groupby("style")[["l1", "ssim", "psnr", "base_l1", "base_ssim"]].mean().round(4); by_style.insert(0, "n", df.groupby("style").size())
print("\n== overall (generated with the true style) vs grayscale-photo baseline ==\n", main)
print("\n== by true style ==\n", by_style); by_style.to_csv(os.path.join(a.out, f"{a.split}_by_style.csv"))
cross_ssim = df.groupby("style")[[f"ssim_c{c}" for c in range(3)]].mean().round(4)
cross_l1 = df.groupby("style")[[f"l1_c{c}" for c in range(3)]].mean().round(4)
print("\n== style-conditioning check: rows = true style, columns = style the generator was told ==")
print("mean SSIM vs ground truth (should peak on the diagonal if the condition is used)\n", cross_ssim)
print("mean L1 vs ground truth (should be lowest on the diagonal)\n", cross_l1)
print("\nmean |difference| between outputs of different styles for the same photo:\n", df[["diff_01", "diff_02", "diff_12"]].mean().round(4))
cross_ssim.to_csv(os.path.join(a.out, f"{a.split}_cross_style_ssim.csv")); cross_l1.to_csv(os.path.join(a.out, f"{a.split}_cross_style_l1.csv"))


def grid(indices, path, title):
    fig, axs = plt.subplots(len(indices), 5, figsize=(10, 2.1 * len(indices))); axs = np.atleast_2d(axs)
    for r, i in enumerate(indices):
        x, y, s = ds[int(i)]
        with torch.no_grad():
            outs = [to01(G(x[None].to(device), torch.tensor([c], device=device)))[0, 0].cpu() for c in range(3)]
        panels = [(to01(x).permute(1, 2, 0).numpy(), f"photo ({ds.names[i]})", None), (to01(y)[0].numpy(), f"ground truth (style {s})", "gray")]
        panels += [(o.numpy(), f"generated, style {c}" + ("  <- true" if c == s else ""), "gray") for c, o in enumerate(outs)]
        for c, (im, t, cm) in enumerate(panels):
            axs[r, c].imshow(im, cmap=cm, vmin=0, vmax=1); axs[r, c].set_title(t, fontsize=7); axs[r, c].axis("off")
    fig.suptitle(title, y=0.995); fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(path, dpi=140); plt.close(fig)


rng = np.random.default_rng(0)
pick = [int(i) for s in range(3) for i in rng.choice(np.where(ds.styles == s)[0], min(3, (ds.styles == s).sum()), replace=False)]
grid(pick, os.path.join(a.out, f"{a.split}_examples.png"), "Examples: photo, ground truth, generated in all three styles")
grid(df.nsmallest(4, "ssim").index.tolist(), os.path.join(a.out, f"{a.split}_failures.png"), "Failure cases (lowest SSIM)")
print("saved to", a.out)
