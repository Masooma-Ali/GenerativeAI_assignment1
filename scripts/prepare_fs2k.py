"""Prepare the FS2K dataset for Task 4.

    python scripts/prepare_fs2k.py --root /kaggle/input/<your-dataset>/FS2K --out /kaggle/working/fs2k

* official train/test split = anno_train.json / anno_test.json (test is never used for training or tuning)
* 15% of the official TRAIN set -> validation, stratified by sketch style, random seed 42
* photos -> RGB 128x128, sketches -> grayscale 128x128; pairing is checked file by file
* writes train.npz, val.npz, test.npz, split_info.json and pairs_check.png (look at it!)
"""
import argparse, json, os, re, sys
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SIZE, SEED = 128, 42
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG")


def key_of(path):
    """(folder number, image number): 'photo1/image0110.jpg' and 'sketch1/sketch0110.png' -> ('1', 110)."""
    folder = re.sub(r"^(photo|sketch)", "", os.path.basename(os.path.dirname(path)))
    m = re.search(r"(\d+)$", os.path.splitext(os.path.basename(path))[0])
    return (folder, int(m.group(1))) if m else None


def index_dir(root):
    idx, ex = {}, {}
    for dp, _, files in os.walk(root):
        for f in sorted(files):
            if f.endswith(EXT):
                p = os.path.join(dp, f); k = key_of(p)
                if k: idx[k] = p; ex.setdefault(os.path.basename(dp), []).append(f)
    return idx, ex


def load_split(records, photo_idx, sketch_idx, name):
    photos, sketches, styles, names, missing = [], [], [], [], []
    for r in tqdm(records, desc=name):
        folder, stem = r["image_name"].split("/")
        k = (re.sub(r"^photo", "", folder), int(re.search(r"(\d+)$", stem).group(1)))
        if k not in photo_idx or k not in sketch_idx:
            missing.append((r["image_name"], k in photo_idx, k in sketch_idx)); continue
        photos.append(np.asarray(Image.open(photo_idx[k]).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)))
        sketches.append(np.asarray(Image.open(sketch_idx[k]).convert("L").resize((SIZE, SIZE), Image.BICUBIC)))
        styles.append(int(r["style"])); names.append(r["image_name"])
    return photos, sketches, styles, names, missing


def main(root, out):
    os.makedirs(out, exist_ok=True)
    photo_idx, pex = index_dir(os.path.join(root, "photo"))
    sketch_idx, sex = index_dir(os.path.join(root, "sketch"))
    print(f"found {len(photo_idx)} photos and {len(sketch_idx)} sketches")
    for d, fl in {**{f"photo/{k}": v for k, v in pex.items()}, **{f"sketch/{k}": v for k, v in sex.items()}}.items():
        print(f"  {d}: {len(fl)} files, e.g. {fl[:3]}")
    train_rec = json.load(open(os.path.join(root, "anno_train.json")))
    test_rec = json.load(open(os.path.join(root, "anno_test.json")))
    res = {}
    for name, rec in [("train_all", train_rec), ("test", test_rec)]:
        ph, sk, st, nm, miss = load_split(rec, photo_idx, sketch_idx, name)
        if miss:
            print(f"\nERROR: {len(miss)} records of '{name}' have no matching photo/sketch file, e.g. {miss[:5]}")
            print("sketch keys look like:", list(sketch_idx)[:5], "| photo keys:", list(photo_idx)[:5])
            sys.exit(1)
        res[name] = (np.stack(ph), np.stack(sk), np.array(st), np.array(nm))

    ph, sk, st, nm = res["train_all"]
    tr_i, va_i = train_test_split(np.arange(len(st)), test_size=0.15, stratify=st, random_state=SEED)
    sets = {"train": tr_i, "val": va_i}
    for name, ii in sets.items():
        np.savez_compressed(os.path.join(out, f"{name}.npz"), photos=ph[ii], sketches=sk[ii], styles=st[ii], names=nm[ii])
    tp, ts, tst, tn = res["test"]
    np.savez_compressed(os.path.join(out, "test.npz"), photos=tp, sketches=ts, styles=tst, names=tn)

    info = {k: {"n": int(len(ii)), "style_counts": np.bincount(st[ii], minlength=3).tolist()} for k, ii in sets.items()}
    info["test"] = {"n": int(len(tst)), "style_counts": np.bincount(tst, minlength=3).tolist()}
    json.dump(info, open(os.path.join(out, "split_info.json"), "w"), indent=2); print(json.dumps(info, indent=2))

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rng = np.random.default_rng(0); pick = rng.choice(len(tr_i), 6, replace=False)
    fig, axs = plt.subplots(2, 6, figsize=(14, 5))
    for c, j in enumerate(pick):
        axs[0, c].imshow(ph[tr_i[j]]); axs[1, c].imshow(sk[tr_i[j]], cmap="gray")
        axs[0, c].set_title(f"{nm[tr_i[j]]}  style {st[tr_i[j]]}", fontsize=7); axs[0, c].axis("off"); axs[1, c].axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(out, "pairs_check.png"), dpi=120)
    print("saved", out, "- open pairs_check.png and confirm each sketch matches the photo above it")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--out", default="fs2k")
    a = ap.parse_args(); main(a.root, a.out)
