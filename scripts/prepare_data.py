"""Download Oxford-IIIT Pet, resize to 128x128 RGB, create the 80/20 split
(seed 42) and the deterministic validation/test corruption manifests.

    python scripts/prepare_data.py --root data
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.corruptions import fixed_test_conditions, sample_train_condition

SIZE, SEED = 128, 42


def to_array(ds):
    return np.stack([np.asarray(img.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC))
                     for img, _ in tqdm(ds, desc="resize")]).astype(np.uint8)


def main(root):
    from torchvision.datasets import OxfordIIITPet
    out = os.path.join(root, "processed"); os.makedirs(out, exist_ok=True)

    trainval = OxfordIIITPet(root, split="trainval", download=True)
    test = OxfordIIITPet(root, split="test", download=True)
    tv, te = to_array(trainval), to_array(test)
    np.save(os.path.join(out, "trainval_128.npy"), tv)
    np.save(os.path.join(out, "test_128.npy"), te)

    # 80/20 split, seed 42 (shared by Tasks 1-3)
    perm = np.random.RandomState(SEED).permutation(len(tv))
    n_train = int(0.8 * len(tv))
    split = {"train": perm[:n_train].tolist(), "val": perm[n_train:].tolist()}
    json.dump(split, open(os.path.join(out, "split.json"), "w"))

    # validation manifest: 1 stored condition per val image (seeded)
    rng = np.random.default_rng(SEED)
    with open(os.path.join(out, "val_manifest.jsonl"), "w") as f:
        for idx in split["val"]:
            f.write(json.dumps({"img_idx": idx, "cond": sample_train_condition(rng, SIZE)}) + "\n")

    # test manifest: 10 fixed conditions per test image (clean + 3x3)
    rng = np.random.default_rng(SEED + 1)
    with open(os.path.join(out, "test_manifest.jsonl"), "w") as f:
        for idx in range(len(te)):
            for cond in fixed_test_conditions(rng, SIZE):
                f.write(json.dumps({"img_idx": idx, "cond": cond}) + "\n")

    print(f"train={len(split['train'])} val={len(split['val'])} test={len(te)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="data")
    main(ap.parse_args().root)
