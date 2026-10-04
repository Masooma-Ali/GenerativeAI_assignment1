"""Figure of classifier mistakes (for the report's failure-case discussion).
    python scripts/show_clf_errors.py --data-root /kaggle/working/data \
        --per-sample /kaggle/working/clf_test_eval/test_per_sample.csv --out /kaggle/working/clf_test_eval/errors.png"""
import argparse, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.corruptions import apply_condition
from src.data.pets import load_manifest

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", default="data"); ap.add_argument("--per-sample", required=True)
ap.add_argument("--out", default="errors.png"); ap.add_argument("--n", type=int, default=8)
a = ap.parse_args()

proc = os.path.join(a.data_root, "processed")
images = np.load(os.path.join(proc, "test_128.npy"))
manifest = load_manifest(os.path.join(proc, "test_manifest.jsonl"))
df = pd.read_csv(a.per_sample)
assert len(df) == len(manifest), "per-sample CSV and manifest must come from the same test split"
err = df[~df.correct]
print("errors:", len(err)); print(err.groupby(["type", "severity", "pred"]).size())

rng = np.random.default_rng(0)
idx = rng.choice(err.index.to_numpy(), min(a.n, len(err)), replace=False)
cols = 4; rows = int(np.ceil(len(idx) / cols))
fig, axs = plt.subplots(rows, cols, figsize=(3 * cols, 3.2 * rows)); axs = np.atleast_1d(axs).ravel()
for ax, i in zip(axs, idx):
    e = manifest[i]; img = apply_condition(images[e["img_idx"]], e["cond"])
    ax.imshow(img); ax.axis("off")
    r = df.loc[i]
    ax.set_title(f"true {r.type}/{r.severity}\npred {r.pred} (p={r[f'p_{r.pred}']:.2f})", fontsize=8)
for ax in axs[len(idx):]: ax.axis("off")
fig.suptitle("Classifier mistakes (test set)"); fig.tight_layout(); fig.savefig(a.out, dpi=140)
print("saved", a.out)
