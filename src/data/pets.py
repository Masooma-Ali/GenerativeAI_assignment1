"""Dataset classes for Tasks 1-3 (uses preprocessed uint8 arrays)."""
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from .corruptions import apply_condition, label_of, sample_train_condition


def _to_tensor(img_u8):
    return torch.from_numpy(img_u8).permute(2, 0, 1).float().div_(255.0)


class PetRestorationDataset(Dataset):
    """
    Training: pass `indices`; corruption is sampled at RUNTIME on every __getitem__
    (nothing is saved to disk).
    Val/test: pass `manifest`; corruption is read from the stored manifest.
    Returns (corrupted, clean, label) as float32 CxHxW in [0,1] and an int label.
    """
    def __init__(self, images, indices=None, manifest=None):
        assert (manifest is None) != (indices is None), "give indices XOR manifest"
        self.images = images                    # N x 128 x 128 x 3 uint8
        self.manifest = manifest                # list of {"img_idx", "cond"} or None
        self.indices = np.asarray(indices) if indices is not None else None

    def __len__(self):
        return len(self.manifest) if self.manifest is not None else len(self.indices)

    def __getitem__(self, i):
        if self.manifest is not None:
            e = self.manifest[i]
            clean, cond = self.images[e["img_idx"]], e["cond"]
        else:
            clean = self.images[self.indices[i]]
            cond = sample_train_condition(np.random.default_rng())   # fresh OS entropy
        return _to_tensor(apply_condition(clean, cond)), _to_tensor(clean), label_of(cond)


def load_manifest(path):
    with open(path) as f:
        return [json.loads(line) for line in f]
