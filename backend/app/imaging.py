"""Image I/O + metrics for the API. Preprocessing mirrors training: RGB, 128x128, PIL bicubic, [0,1] float."""
import base64
import io

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

SIZE = 128
MAX_BYTES = 10 * 1024 * 1024
ALLOWED_FORMATS = {"PNG", "JPEG", "BMP", "WEBP", "TIFF"}


class BadImage(ValueError):
    pass


def decode_upload(data: bytes, mode="RGB"):
    """bytes -> (HxWxC uint8 resized to 128x128, original (w, h))."""
    if not data:
        raise BadImage("Empty file.")
    if len(data) > MAX_BYTES:
        raise BadImage(f"File too large ({len(data) / 1e6:.1f} MB, limit {MAX_BYTES // 1024 // 1024} MB).")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as e:
        raise BadImage("File is not a readable image.") from e
    if img.format not in ALLOWED_FORMATS:
        raise BadImage(f"Unsupported format {img.format}; use one of {sorted(ALLOWED_FORMATS)}.")
    orig = img.size
    img = img.convert(mode).resize((SIZE, SIZE), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8), orig


def to_tensor(img):
    """HxWx3 uint8 -> 1x3xHxW float32 in [0,1]."""
    return (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def from_tensor(t):
    """CxHxW (or 1xCxHxW) float in [0,1] -> HxWxC uint8 (C=1 squeezed to HxW)."""
    t = np.asarray(t)
    if t.ndim == 4:
        t = t[0]
    img = (np.clip(t, 0, 1).transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)
    return img[..., 0] if img.shape[2] == 1 else img


def to_png_b64(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def error_map(a, b):
    """Mean absolute per-pixel error, colour-mapped (dark = 0, bright = large)."""
    err = np.abs(a.astype(np.float32) - b.astype(np.float32)).mean(2)
    err = np.clip(err * 3.0, 0, 255).astype(np.uint8)            # x3 gain so small errors are visible
    return cv2.cvtColor(cv2.applyColorMap(err, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def ssim(a, b):
    """Standard SSIM (11x11 Gaussian, sigma 1.5), averaged over RGB channels."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    vals = []
    for ch in range(a.shape[2]):
        x, y = a[..., ch].astype(np.float64), b[..., ch].astype(np.float64)
        blur = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
        mx, my = blur(x), blur(y)
        sxx, syy, sxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
        m = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
        vals.append(m.mean())
    return float(np.mean(vals))


def metrics(clean, img):
    v = psnr(clean, img)
    return {"psnr": None if np.isinf(v) else round(v, 2), "ssim": round(ssim(clean, img), 4)}   # None = identical
