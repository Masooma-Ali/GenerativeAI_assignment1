"""Evaluate the classifier: accuracy, macro P/R/F1, per-class metrics, normalised confusion matrix,
and accuracy per corruption type x severity.
    python scripts/eval_clf.py --ckpt clf_final/best.pt --split val
    python scripts/eval_clf.py --ckpt clf_final/best.pt --split test --final      # only for the final report"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.corruptions import CLASSES
from src.data.pets import PetRestorationDataset, load_manifest
from src.train_clf import build_clf

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--data-root", default="data")
ap.add_argument("--split", choices=["val", "test"], default="val"); ap.add_argument("--final", action="store_true")
ap.add_argument("--out", default="clf_eval")
a = ap.parse_args()
if a.split == "test" and not a.final:
    sys.exit("The test set must stay untouched until final evaluation. Re-run with --final when you are done tuning.")

device = "cuda" if torch.cuda.is_available() else "cpu"
proc = os.path.join(a.data_root, "processed")
images = np.load(os.path.join(proc, "trainval_128.npy" if a.split == "val" else "test_128.npy"))
manifest = load_manifest(os.path.join(proc, f"{a.split}_manifest.jsonl"))
ds = PetRestorationDataset(images, manifest=manifest)
ck = torch.load(a.ckpt, map_location=device)
model = build_clf(ck["cfg"]).to(device); model.load_state_dict(ck["model"]); model.eval()

probs, ys = [], []
with torch.no_grad():
    for x, _, y in DataLoader(ds, batch_size=256, shuffle=False, num_workers=2):
        probs.append(torch.softmax(model(x.to(device)), 1).cpu().numpy()); ys += y.tolist()
probs, ys = np.concatenate(probs), np.array(ys); pred = probs.argmax(1)
os.makedirs(a.out, exist_ok=True)

rep = classification_report(ys, pred, target_names=CLASSES, output_dict=True, digits=4)
rep["accuracy"] = float(accuracy_score(ys, pred))
json.dump(rep, open(os.path.join(a.out, f"{a.split}_metrics.json"), "w"), indent=2)
print(f"accuracy {rep['accuracy']:.4f} | macro P {rep['macro avg']['precision']:.4f} "
      f"R {rep['macro avg']['recall']:.4f} F1 {rep['macro avg']['f1-score']:.4f}")
print(pd.DataFrame({c: rep[c] for c in CLASSES}).T.round(4))

cm = confusion_matrix(ys, pred, normalize="true")
fig, ax = plt.subplots(figsize=(5, 4.5)); im = ax.imshow(cm, vmin=0, vmax=1, cmap="Blues")
ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES, rotation=30); ax.set_yticks(range(4)); ax.set_yticklabels(CLASSES)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", color="white" if cm[i, j] > .5 else "black")
ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(f"Normalised confusion matrix ({a.split})")
fig.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(a.out, f"{a.split}_confusion.png"), dpi=150); plt.close(fig)

df = pd.DataFrame({"type": [e["cond"]["type"] for e in manifest], "severity": [e["cond"]["severity"] for e in manifest],
                   "correct": ys == pred, "pred": [CLASSES[p] for p in pred]})
for i, c in enumerate(CLASSES): df[f"p_{c}"] = probs[:, i]
df.to_csv(os.path.join(a.out, f"{a.split}_per_sample.csv"), index=False)
acc = df.groupby(["type", "severity"])["correct"].mean().round(4); acc.to_csv(os.path.join(a.out, f"{a.split}_acc_by_severity.csv"))
print("\naccuracy by type x severity:\n", acc)
