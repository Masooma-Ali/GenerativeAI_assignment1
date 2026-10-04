"""Task 3 soft mixture-of-experts.

    w = softmax(gate(x) / T)                      w = [w_clean, w_salt, w_blur, w_occlusion]
    x_hat = w0*x + w1*E_salt(x) + w2*E_blur(x) + w3*E_occ(x)       (differentiable)
The gate is the Task 2 classifier, the experts are the Task 2 specialists, branch 0 is the identity."""
import torch
import torch.nn as nn


class SoftMoE(nn.Module):
    def __init__(self, gate, experts, temperature=1.0):
        super().__init__()
        self.gate, self.experts = gate, nn.ModuleList(experts)
        self.temperature = float(temperature)
        self.freeze_experts = False                  # True during the gate warm-up stage

    def forward(self, x):
        logits = self.gate(x)
        w = torch.softmax(logits / self.temperature, dim=1)
        if self.freeze_experts:
            with torch.no_grad():
                outs = [e(x) for e in self.experts]
        else:
            outs = [e(x) for e in self.experts]
        branches = torch.stack([x] + outs, dim=1)                    # B x 4 x 3 x H x W
        recon = (w[:, :, None, None, None] * branches).sum(1)
        return recon, w, logits
