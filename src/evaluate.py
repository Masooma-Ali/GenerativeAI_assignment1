"""Generic evaluation: works for Task 1 (single AE) and later Tasks 2/3 by passing
a different `predict_fn` (tensor batch -> restored batch)."""
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .losses import batch_metrics


@torch.no_grad()
def evaluate(predict_fn, ds, manifest, device, batch_size=256, num_workers=2):
    """Returns a per-sample DataFrame (same order as `manifest`) with the corruption
    type/severity and PSNR/SSIM of the corrupted INPUT and of the RESTORED output."""
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    rows = {"psnr_in": [], "ssim_in": [], "psnr_out": [], "ssim_out": []}
    for x, y, _ in dl:
        x, y = x.to(device), y.to(device)
        p_in, s_in = batch_metrics(x, y)
        p_out, s_out = batch_metrics(predict_fn(x).clamp(0, 1), y)
        for k, v in zip(rows, (p_in, s_in, p_out, s_out)):
            rows[k] += v.cpu().tolist()
    df = pd.DataFrame(rows)
    df["type"] = [e["cond"]["type"] for e in manifest]
    df["severity"] = [e["cond"]["severity"] for e in manifest]
    df["img_idx"] = [e["img_idx"] for e in manifest]
    return df


def summarize(df, by=("type", "severity")):
    cols = ["psnr_in", "ssim_in", "psnr_out", "ssim_out"]
    out = df.groupby(list(by))[cols].mean().round(4)
    out.insert(0, "n", df.groupby(list(by)).size())
    return out
