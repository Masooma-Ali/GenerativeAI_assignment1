import json, os
import numpy as np
from src.data.pets import load_manifest


def load_pets(data_root):
    proc = os.path.join(data_root, "processed")
    images = np.load(os.path.join(proc, "trainval_128.npy"))
    split = json.load(open(os.path.join(proc, "split.json")))
    val_manifest = load_manifest(os.path.join(proc, "val_manifest.jsonl"))
    return images, split, val_manifest, proc
