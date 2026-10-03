"""Corruption library for Tasks 1-3.

Everything here works on HxWx3 uint8 numpy arrays and is DETERMINISTIC given a
`cond` dict (all randomness, including the salt-and-pepper mask, is derived
from cond["seed"]). That is what lets us store validation/test manifests.

cond schema (JSON-serialisable):
  {"type": "clean"|"salt_pepper"|"blur"|"occlusion",
   "severity": "low"|"medium"|"high"|"sampled"|"none",
   "p": float,                       # salt_pepper
   "kernel": int, "sigma": float,    # blur
   "boxes": [[y0,x0,y1,x1], ...],    # occlusion
   "seed": int}
"""
import cv2
import numpy as np

CLASSES = ["clean", "salt_pepper", "blur", "occlusion"]   # label = index
SEVERITIES = ["low", "medium", "high"]

# Fixed final-test levels from the assignment brief
TEST_SALT_P = [0.03, 0.08, 0.15]
TEST_BLUR = [(3, 0.7), (5, 1.5), (7, 2.5)]                # (kernel, sigma)
TEST_OCC = [(1, 0.10), (2, 0.20), (3, 0.35)]              # (n_rects, coverage)


def label_of(cond):
    return CLASSES.index(cond["type"])


# ---------------------------------------------------------------- occlusion
def sample_boxes(rng, n_rects, coverage, size=128, tries=50):
    """n black rectangles whose areas sum to ~coverage * size^2.
    Area is split randomly; aspect ratio h/w in [0.5, 2]. Positions are uniform,
    but we retry (up to `tries` times) to avoid overlap so the true coverage
    stays close to the target. Use occlusion_coverage() for the exact value."""
    total = coverage * size * size
    fractions = rng.dirichlet(np.full(n_rects, 4.0))      # avoids tiny slivers
    mask = np.zeros((size, size), bool)
    boxes = []
    for a in fractions * total:
        ar = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))
        h = int(np.clip(round(np.sqrt(a * ar)), 1, size))
        w = int(np.clip(round(np.sqrt(a / ar)), 1, size))
        best, best_ov = None, None
        for _ in range(tries):
            y0 = int(rng.integers(0, size - h + 1))
            x0 = int(rng.integers(0, size - w + 1))
            ov = int(mask[y0:y0 + h, x0:x0 + w].sum())
            if best is None or ov < best_ov:
                best, best_ov = [y0, x0, y0 + h, x0 + w], ov
            if ov == 0:
                break
        boxes.append(best)
        mask[best[0]:best[2], best[1]:best[3]] = True
    return boxes


def occlusion_coverage(boxes, size=128):
    m = np.zeros((size, size), bool)
    for y0, x0, y1, x1 in boxes:
        m[y0:y1, x0:x1] = True
    return float(m.mean())


# ------------------------------------------------------------ condition samplers
def sample_condition(rng, t, size=128):
    """Random severity for a GIVEN corruption type t (used by the balanced classifier
    sampler and by the Task 2 specialists)."""
    cond = {"type": t, "severity": "sampled", "seed": int(rng.integers(0, 2**31 - 1))}
    if t == "salt_pepper":
        cond["p"] = float(rng.uniform(0.02, 0.15))
    elif t == "blur":
        cond["kernel"] = int(rng.choice([3, 5, 7]))
        cond["sigma"] = float(rng.uniform(0.5, 2.5))
    elif t == "occlusion":
        n = int(rng.integers(1, 4))
        cond["boxes"] = sample_boxes(rng, n, float(rng.uniform(0.10, 0.35)), size)
    return cond


def sample_train_condition(rng, size=128):
    """Training: equal-probability choice of the 4 conditions + random severity."""
    return sample_condition(rng, CLASSES[int(rng.integers(0, 4))], size)


def fixed_test_conditions(rng, size=128):
    """Clean + 3 severities x 3 corruptions = 10 conditions for one test image."""
    def seed():
        return int(rng.integers(0, 2**31 - 1))
    out = [{"type": "clean", "severity": "none", "seed": seed()}]
    for sev, p in zip(SEVERITIES, TEST_SALT_P):
        out.append({"type": "salt_pepper", "severity": sev, "p": p, "seed": seed()})
    for sev, (k, s) in zip(SEVERITIES, TEST_BLUR):
        out.append({"type": "blur", "severity": sev, "kernel": k, "sigma": s, "seed": seed()})
    for sev, (n, c) in zip(SEVERITIES, TEST_OCC):
        out.append({"type": "occlusion", "severity": sev,
                    "boxes": sample_boxes(rng, n, c, size), "seed": seed()})
    return out


# ------------------------------------------------------------------- apply
def apply_condition(img, cond):
    """img: HxWx3 uint8 -> corrupted HxWx3 uint8 (new array)."""
    t = cond["type"]
    out = img.copy()
    if t == "clean":
        return out
    if t == "salt_pepper":
        r = np.random.default_rng(cond["seed"])
        h, w = img.shape[:2]
        hit = r.random((h, w)) < cond["p"]                 # pixel selected
        white = r.random((h, w)) < 0.5                     # 50/50 black or white
        out[hit & white] = 255
        out[hit & ~white] = 0
        return out
    if t == "blur":
        k = cond["kernel"]
        return cv2.GaussianBlur(out, (k, k), sigmaX=cond["sigma"], sigmaY=cond["sigma"])
    if t == "occlusion":
        for y0, x0, y1, x1 in cond["boxes"]:
            out[y0:y1, x0:x1] = 0
        return out
    raise ValueError(f"unknown corruption type {t}")