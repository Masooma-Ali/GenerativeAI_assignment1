"""FS2K paired dataset for Task 4. Photos and sketches are scaled to [-1, 1]."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class FS2KDataset(Dataset):
    """Returns (photo 3x128x128, sketch 1x128x128, style int). Spatial augmentation (flip, random crop+resize)
    is applied to the STACKED 4-channel tensor, so photo and sketch always receive the identical transform."""
    def __init__(self, npz_path, augment=False):
        d = np.load(npz_path, allow_pickle=True)
        self.photos, self.sketches, self.styles, self.names = d["photos"], d["sketches"], d["styles"], d["names"]
        self.augment = augment

    def __len__(self):
        return len(self.styles)

    def __getitem__(self, i):
        p = torch.from_numpy(self.photos[i]).permute(2, 0, 1).float() / 127.5 - 1
        s = torch.from_numpy(self.sketches[i])[None].float() / 127.5 - 1
        if self.augment:
            pair = torch.cat([p, s], 0)
            if torch.rand(1).item() < 0.5:
                pair = pair.flip(-1)
            size = int(round(128 * float(torch.empty(1).uniform_(0.8, 1.0))))
            if size < 128:
                y0 = torch.randint(0, 128 - size + 1, (1,)).item(); x0 = torch.randint(0, 128 - size + 1, (1,)).item()
                pair = F.interpolate(pair[:, y0:y0 + size, x0:x0 + size][None], size=(128, 128),
                                     mode="bilinear", align_corners=False)[0]
            p, s = pair[:3], pair[3:]
        return p, s, int(self.styles[i])
