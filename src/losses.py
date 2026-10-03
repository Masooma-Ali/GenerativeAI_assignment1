import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class RestorationLoss(nn.Module):
    """L = alpha * L1 + (1 - alpha) * (1 - SSIM)   (images in [0,1])"""
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        l1 = F.l1_loss(pred, target)
        s = ssim(pred, target, data_range=1.0, size_average=True)
        return self.alpha * l1 + (1 - self.alpha) * (1 - s), l1.detach(), s.detach()


@torch.no_grad()
def batch_metrics(pred, target):
    """Per-image PSNR (dB, capped at 50) and SSIM. Returns two tensors of shape (N,)."""
    mse = ((pred - target) ** 2).flatten(1).mean(1).clamp_min(1e-5)
    psnr = -10 * torch.log10(mse)
    s = ssim(pred, target, data_range=1.0, size_average=False)
    return psnr, s


def combined_score(psnr, ssim_val):
    """Single validation objective (higher = better): half structural similarity,
    half pixel fidelity (PSNR normalised to [0,1] with a 40 dB ceiling).
    Reused as the Optuna objective."""
    return 0.5 * ssim_val + 0.5 * min(psnr, 40.0) / 40.0
