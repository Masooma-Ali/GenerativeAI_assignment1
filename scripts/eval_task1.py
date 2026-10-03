"""Evaluate a trained Task 1 model per corruption type and severity, and make
example / failure-case figures with absolute error maps.

    python scripts/eval_task1.py --ckpt task1/best.pt --split val          # safe during development
    python scripts/eval_task1.py --ckpt task1/best.pt --split test --final # ONLY for the final report
"""
import argparse, json, os, sys
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.pets import PetRestorationDataset, load_manifest
from src.evaluate import evaluate, summarize
from src.train_ae import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--data-root", default="data")
ap.add_argument("--split", choices=["val", "test"], default="val"); ap.add_argument("--final", action="store_true")
ap.add_argument("--out", default="task1_eval")
a = ap.parse_args()
if a.split == "test" and not a.final:
    sys.exit("The test set must stay untouched until final evaluation. Re-run with --final when you are done tuning.")

device = "cuda" if torch.cuda.is_available() else "cpu"
proc = os.path.join(a.data_root, "processed")
images = np.load(os.path.join(proc, "trainval_128.npy" if a.split == "val" else "test_128.npy"))
manifest = load_manifest(os.path.join(proc, f"{a.split}_manifest.jsonl"))
ds = PetRestorationDataset(images, manifest=manifest)

ck = torch.load(a.ckpt, map_location=device)
model = build_model(ck["cfg"]).to(device); model.load_state_dict(ck["model"]); model.eval()

df = evaluate(lambda x: model(x), ds, manifest, device)
os.makedirs(a.out, exist_ok=True)
df.to_csv(os.path.join(a.out, f"{a.split}_per_sample.csv"), index=False)
by_type = summarize(df, by=("type",)); by_sev = summarize(df)
by_type.to_csv(os.path.join(a.out, f"{a.split}_by_type.csv")); by_sev.to_csv(os.path.join(a.out, f"{a.split}_by_type_severity.csv"))
print("\n== by corruption type ==\n", by_type, "\n\n== by type and severity ==\n", by_sev)


def figure(indices, path, title):
    fig, ax = plt.subplots(len(indices), 4, figsize=(8, 2 * len(indices)))
    ax = np.atleast_2d(ax)
    for r, i in enumerate(indices):
        x, y, _ = ds[i]
        with torch.no_grad():
            p = model(x[None].to(device)).clamp(0, 1)[0].cpu()
        err = (p - y).abs().mean(0)
        row = df.iloc[i]
        for c, (im, name) in enumerate([(y, "clean"), (x, f"{row.type}/{row.severity}"),
                                        (p, f"restored {row.ssim_out:.2f}"), (err, "abs error")]):
            ax[r, c].imshow(im.permute(1, 2, 0).numpy() if im.ndim == 3 else im.numpy(),
                            cmap=None if im.ndim == 3 else "inferno", vmin=0, vmax=None if im.ndim == 3 else 0.5)
            ax[r, c].set_title(name, fontsize=7); ax[r, c].axis("off")
    fig.suptitle(title, y=0.995); fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(path, dpi=130); plt.close(fig)


rng = np.random.default_rng(0)
picks = []
for t in ["salt_pepper", "blur", "occlusion"]:
    ids = df.index[df.type == t].to_numpy()
    picks += rng.choice(ids, 4, replace=False).tolist()           # 12 representative examples
figure(picks, os.path.join(a.out, f"{a.split}_examples.png"), "Representative examples")
worst = df[df.type != "clean"].nsmallest(4, "ssim_out").index.tolist()   # 4 failure cases
figure(worst, os.path.join(a.out, f"{a.split}_failures.png"), "Failure cases (lowest SSIM)")
print("saved figures to", a.out)