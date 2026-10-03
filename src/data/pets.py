"""Dataset classes for Tasks 1-3 (uses preprocessed uint8 arrays)."""
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from .corruptions import CLASSES, apply_condition, label_of, sample_condition, sample_train_condition


def _to_tensor(img_u8):
    return torch.from_numpy(img_u8).permute(2, 0, 1).float().div_(255.0)


class PetRestorationDataset(Dataset):
    """
    Training: pass `indices`; corruption is sampled at RUNTIME on every __getitem__
    (nothing is saved to disk).
    Val/test: pass `manifest`; corruption is read from the stored manifest.
    Returns (corrupted, clean, label) as float32 CxHxW in [0,1] and an int label.
    """
    def __init__(self, images, indices=None, manifest=None, types=None):
        """types: optional list of corruption types to keep (Task 2 specialists), e.g. ["blur"]."""
        assert (manifest is None) != (indices is None), "give indices XOR manifest"
        self.images = images                    # N x 128 x 128 x 3 uint8
        if manifest is not None and types is not None:
            manifest = [e for e in manifest if e["cond"]["type"] in types]
        self.manifest = manifest                # list of {"img_idx", "cond"} or None
        self.indices = np.asarray(indices) if indices is not None else None
        self.types = types

    def __len__(self):
        return len(self.manifest) if self.manifest is not None else len(self.indices)

    def __getitem__(self, i):
        if self.manifest is not None:
            e = self.manifest[i]
            clean, cond = self.images[e["img_idx"]], e["cond"]
        else:
            clean = self.images[self.indices[i]]
            rng = np.random.default_rng()                            # fresh OS entropy
            cond = (sample_condition(rng, self.types[int(rng.integers(len(self.types)))])
                    if self.types else sample_train_condition(rng))
        return _to_tensor(apply_condition(clean, cond)), _to_tensor(clean), label_of(cond)


def load_manifest(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


class BalancedCorruptionDataset(Dataset):
    """For the Task 2 classifier. Length = 4 * n_images; index i -> (image i//4, class i%4).
    Use with BalancedBatchSampler so EVERY batch holds exactly batch_size/4 per class.
    Returns (corrupted, clean, label) like PetRestorationDataset."""
    def __init__(self, images, indices):
        self.images, self.indices = images, np.asarray(indices)

    def __len__(self):
        return 4 * len(self.indices)

    def __getitem__(self, i):
        c = i % 4
        clean = self.images[self.indices[i // 4]]
        cond = sample_condition(np.random.default_rng(), CLASSES[c])
        return _to_tensor(apply_condition(clean, cond)), _to_tensor(clean), c


class BalancedBatchSampler:
    """Yields lists of indices; each batch has batch_size/4 samples of every class."""
    def __init__(self, n_images, batch_size):
        assert batch_size % 4 == 0, "batch_size must be a multiple of 4"
        self.n, self.per = n_images, batch_size // 4

    def __len__(self):
        return self.n // self.per

    def __iter__(self):
        rng = np.random.default_rng()
        streams = [rng.permutation(self.n) for _ in range(4)]    # a fresh image order per class
        for b in range(len(self)):
            yield [int(streams[c][k]) * 4 + c for c in range(4)
                   for k in range(b * self.per, (b + 1) * self.per)]