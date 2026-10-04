"""Task 2 hard-routed restoration pipeline (PyTorch reference implementation).

    p = softmax(classifier(x));  k = argmax p
    k == clean      -> identity bypass (output = input, no expert runs)
    k == salt/blur/occlusion -> the matching specialist autoencoder
"""
import os
import torch

from .data.corruptions import CLASSES
from .train_ae import build_model
from .train_clf import build_clf


def _load(path, builder, device):
    ck = torch.load(path, map_location=device)
    m = builder(ck["cfg"]).to(device); m.load_state_dict(ck["model"]); m.eval()
    return m


class HardRouter:
    def __init__(self, clf_ckpt, spec_dir, device="cpu"):
        self.device = device
        self.clf = _load(clf_ckpt, build_clf, device)
        self.experts = {name: _load(os.path.join(spec_dir, name, "best.pt"), build_model, device)
                        for name in CLASSES[1:]}                       # salt_pepper, blur, occlusion

    @torch.no_grad()
    def __call__(self, x, mode="predicted", true_labels=None):
        """mode='predicted': classifier chooses the expert (operational system).
           mode='oracle'   : the known corruption label chooses the expert (upper bound).
        Returns (restored, probs or None, route) with route in {0..3} = index into CLASSES."""
        probs = None
        if mode == "predicted":
            probs = torch.softmax(self.clf(x), 1); route = probs.argmax(1)
        else:
            route = true_labels.to(x.device)
        out = x.clone()                                                # clean -> identity bypass
        for k, name in enumerate(CLASSES):
            sel = route == k
            if k > 0 and sel.any():
                out[sel] = self.experts[name](x[sel]).clamp(0, 1)
        return out, probs, route
