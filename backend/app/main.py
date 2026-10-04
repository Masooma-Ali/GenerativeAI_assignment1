"""FastAPI backend for the four assignment workspaces.

Run from the repository root:
    uvicorn backend.app.main:app --reload --port 8000

Endpoints
    GET  /api/health                 server + model status
    GET  /api/samples                clean sample images (backend/samples/)
    GET  /api/samples/{name}
    POST /api/corrupt                preview a runtime corruption
    POST /api/restore/universal      Task 1
    POST /api/restore/hard           Task 2
    POST /api/restore/soft           Task 3
    POST /api/sketch                 Task 4
"""
import json
import os
import platform
import sys
import threading
import time
from typing import Optional

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .imaging import BadImage, decode_upload, error_map, from_tensor, metrics, to_png_b64, to_tensor
from .models import MODEL_FILES, REPO_ROOT, TASK_MODELS, ModelUnavailable, registry

sys.path.insert(0, REPO_ROOT)
from src.data.corruptions import (CLASSES, SEVERITIES, TEST_BLUR, TEST_OCC, TEST_SALT_P,  # noqa: E402
                                  apply_condition, occlusion_coverage, sample_boxes, sample_condition)

SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
EXPERT_LABELS = {"clean": "Identity bypass", "salt_pepper": "Salt-and-pepper expert",
                 "blur": "Blur expert", "occlusion": "Occlusion expert"}
STARTED = time.time()

app = FastAPI(title="GenAI Assignment 1 API", version="1.0")
origins = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def _warmup():
    if os.environ.get("WARMUP", "1") == "1":
        threading.Thread(target=registry.warmup, daemon=True).start()


# ----------------------------------------------------------------------------- helpers
def _bad(msg, code=400):
    raise HTTPException(status_code=code, detail=msg)


async def _read(file: UploadFile, mode="RGB"):
    if file is None:
        _bad("No image uploaded.")
    if file.content_type and not file.content_type.startswith("image/"):
        _bad(f"Expected an image file, got content type '{file.content_type}'.")
    try:
        return decode_upload(await file.read(), mode)
    except BadImage as e:
        _bad(str(e))


def build_condition(corruption, severity, seed, p=None, kernel=None, sigma=None, n_rects=None, coverage=None):
    """Turn form fields into a corruption cond dict (same schema as the training manifests)."""
    if corruption not in CLASSES:
        _bad(f"corruption must be one of {['none'] + CLASSES}.")
    rng = np.random.default_rng(seed)
    if corruption == "clean":
        return {"type": "clean", "severity": "none", "seed": int(seed)}
    if severity == "random":
        cond = sample_condition(rng, corruption)
        cond["seed"] = int(seed)
        return cond
    cond = {"type": corruption, "severity": severity, "seed": int(seed)}
    if severity in SEVERITIES:
        i = SEVERITIES.index(severity)
        if corruption == "salt_pepper":
            cond["p"] = TEST_SALT_P[i]
        elif corruption == "blur":
            cond["kernel"], cond["sigma"] = TEST_BLUR[i]
        else:
            n, c = TEST_OCC[i]
            cond["boxes"] = sample_boxes(rng, n, c)
    elif severity == "custom":
        if corruption == "salt_pepper":
            if p is None or not 0 < p < 1:
                _bad("custom salt-and-pepper needs 0 < p < 1.")
            cond["p"] = float(p)
        elif corruption == "blur":
            if kernel not in (3, 5, 7, 9, 11) or sigma is None or not 0 < sigma <= 10:
                _bad("custom blur needs kernel in {3,5,7,9,11} and 0 < sigma <= 10.")
            cond["kernel"], cond["sigma"] = int(kernel), float(sigma)
        else:
            if n_rects not in (1, 2, 3) or coverage is None or not 0 < coverage <= 0.6:
                _bad("custom occlusion needs n_rects in {1,2,3} and 0 < coverage <= 0.6.")
            cond["boxes"] = sample_boxes(rng, int(n_rects), float(coverage))
    else:
        _bad(f"severity must be one of {SEVERITIES + ['random', 'custom']}.")
    return cond


def describe(cond):
    if cond is None:
        return None
    d = dict(cond)
    if cond["type"] == "occlusion":
        d["n_rects"] = len(cond["boxes"])
        d["coverage"] = round(occlusion_coverage(cond["boxes"]), 3)
    if "p" in d:
        d["p"] = round(d["p"], 4)
    if "sigma" in d:
        d["sigma"] = round(d["sigma"], 3)
    return d


async def prepare(file, corruption, severity, seed, p, kernel, sigma, n_rects, coverage):
    """Upload -> (model input uint8, clean reference or None, cond or None, meta)."""
    t0 = time.perf_counter()
    img, orig = await _read(file)
    if corruption in (None, "", "none"):
        clean, cond, inp = None, None, img
    else:
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31 - 1))
        cond = build_condition(corruption, severity, seed, p, kernel, sigma, n_rects, coverage)
        clean, inp = img, apply_condition(img, cond)
    return inp, clean, cond, {"original_size": list(orig), "preprocess_ms": (time.perf_counter() - t0) * 1000}


def run_model(key, feeds):
    try:
        return registry.run(key, feeds)
    except ModelUnavailable as e:
        _bad(str(e), 503)


def common_response(inp, out, clean, cond, meta, inference_ms, t_start):
    res = {
        "input_image": to_png_b64(inp),
        "output_image": to_png_b64(out),
        "corruption": describe(cond),
        "original_size": meta["original_size"],
        "processed_size": [128, 128],
    }
    if clean is not None:
        res["clean_image"] = to_png_b64(clean)
        res["error_map"] = to_png_b64(error_map(clean, out))
        res["error_map_reference"] = "clean target"
        res["metrics"] = {"input": metrics(clean, inp), "output": metrics(clean, out)}
    else:
        res["error_map"] = to_png_b64(error_map(inp, out))
        res["error_map_reference"] = "input (no clean target available)"
    res["timing"] = {"preprocess_ms": round(meta["preprocess_ms"], 1), "inference_ms": round(inference_ms, 1),
                     "total_ms": round((time.perf_counter() - t_start) * 1000, 1)}
    return res


# ----------------------------------------------------------------------------- routes
@app.get("/api/health")
def health():
    report2 = os.path.join(registry.model_dir, "task2_onnx_report.json")
    task2_report = json.load(open(report2)) if os.path.isfile(report2) else None
    models = {}
    for k in MODEL_FILES:
        models[k] = registry.info(k)
        if task2_report and k in ("classifier", "salt_pepper", "blur", "occlusion") and k in task2_report:
            models[k]["export_report"] = task2_report[k]
    return {
        "status": "ok",
        "uptime_s": round(time.time() - STARTED, 1),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "model_dir": registry.model_dir,
        "tasks": {t: all(registry.available(k) for k in ks) for t, ks in TASK_MODELS.items()},
        "models": models,
        "task2_pipeline_check": task2_report.get("pipeline") if task2_report else None,
    }


@app.get("/api/samples")
def list_samples():
    if not os.path.isdir(SAMPLES_DIR):
        return {"samples": []}
    names = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(IMG_EXT))
    return {"samples": [{"name": n, "url": f"/api/samples/{n}"} for n in names]}


@app.get("/api/samples/{name}")
def get_sample(name: str):
    path = os.path.join(SAMPLES_DIR, os.path.basename(name))
    if not (os.path.isfile(path) and path.lower().endswith(IMG_EXT)):
        _bad("Sample not found.", 404)
    return FileResponse(path)


@app.post("/api/corrupt")
async def corrupt(file: UploadFile = File(...), corruption: str = Form(...), severity: str = Form("medium"),
                  seed: Optional[int] = Form(None), p: Optional[float] = Form(None),
                  kernel: Optional[int] = Form(None), sigma: Optional[float] = Form(None),
                  n_rects: Optional[int] = Form(None), coverage: Optional[float] = Form(None)):
    inp, clean, cond, _ = await prepare(file, corruption, severity, seed, p, kernel, sigma, n_rects, coverage)
    res = {"input_image": to_png_b64(inp), "corruption": describe(cond)}
    if clean is not None:
        res["metrics"] = {"input": metrics(clean, inp)}
    return res


@app.post("/api/restore/universal")
async def restore_universal(file: UploadFile = File(...), corruption: str = Form("none"),
                            severity: str = Form("medium"), seed: Optional[int] = Form(None),
                            p: Optional[float] = Form(None), kernel: Optional[int] = Form(None),
                            sigma: Optional[float] = Form(None), n_rects: Optional[int] = Form(None),
                            coverage: Optional[float] = Form(None)):
    t = time.perf_counter()
    inp, clean, cond, meta = await prepare(file, corruption, severity, seed, p, kernel, sigma, n_rects, coverage)
    (restored,), ms = run_model("universal", {"input": to_tensor(inp)})
    out = from_tensor(restored)
    res = common_response(inp, out, clean, cond, meta, ms, t)
    res["model"] = MODEL_FILES["universal"]
    return res


@app.post("/api/restore/hard")
async def restore_hard(file: UploadFile = File(...), corruption: str = Form("none"),
                       severity: str = Form("medium"), seed: Optional[int] = Form(None),
                       p: Optional[float] = Form(None), kernel: Optional[int] = Form(None),
                       sigma: Optional[float] = Form(None), n_rects: Optional[int] = Form(None),
                       coverage: Optional[float] = Form(None), mode: str = Form("predicted")):
    if mode not in ("predicted", "oracle"):
        _bad("mode must be 'predicted' or 'oracle'.")
    t = time.perf_counter()
    inp, clean, cond, meta = await prepare(file, corruption, severity, seed, p, kernel, sigma, n_rects, coverage)
    if mode == "oracle" and cond is None:
        _bad("Oracle routing needs a known corruption label: apply a corruption in the app instead of 'none'.")
    x = to_tensor(inp)
    (probs,), clf_ms = run_model("classifier", {"input": x})
    probs = probs[0].astype(float)
    predicted = CLASSES[int(probs.argmax())]
    route = cond["type"] if mode == "oracle" else predicted
    if route == "clean":
        out, exp_ms = inp.copy(), 0.0                       # identity bypass: no expert runs
    else:
        (restored,), exp_ms = run_model(route, {"input": x})
        out = from_tensor(restored)
    res = common_response(inp, out, clean, cond, meta, clf_ms + exp_ms, t)
    res.update({
        "mode": mode,
        "probabilities": {c: round(float(v), 5) for c, v in zip(CLASSES, probs)},
        "predicted": predicted,
        "confidence": round(float(probs.max()), 5),
        "true_label": cond["type"] if cond else None,
        "routed_to": route,
        "selected_expert": EXPERT_LABELS[route],
        "expert_model": None if route == "clean" else MODEL_FILES[route],
        "misrouted": (cond is not None and predicted != cond["type"]),
    })
    res["timing"].update({"classifier_ms": round(clf_ms, 1), "expert_ms": round(exp_ms, 1)})
    return res


@app.post("/api/restore/soft")
async def restore_soft(file: UploadFile = File(...), corruption: str = Form("none"),
                       severity: str = Form("medium"), seed: Optional[int] = Form(None),
                       p: Optional[float] = Form(None), kernel: Optional[int] = Form(None),
                       sigma: Optional[float] = Form(None), n_rects: Optional[int] = Form(None),
                       coverage: Optional[float] = Form(None)):
    t = time.perf_counter()
    inp, clean, cond, meta = await prepare(file, corruption, severity, seed, p, kernel, sigma, n_rects, coverage)
    (restored, weights), ms = run_model("soft_moe", {"input": to_tensor(inp)})
    out = from_tensor(restored)
    w = weights[0].astype(float)
    res = common_response(inp, out, clean, cond, meta, ms, t)
    order = np.argsort(-w)
    res.update({
        "weights": {c: round(float(v), 5) for c, v in zip(CLASSES, w)},
        "dominant": CLASSES[int(order[0])],
        "dominant_label": EXPERT_LABELS[CLASSES[int(order[0])]],
        "ranking": [CLASSES[int(i)] for i in order],
        "routing_entropy": round(float(-(w * np.log(w + 1e-12)).sum() / np.log(4)), 4),   # 0 = one-hot, 1 = uniform
        "true_label": cond["type"] if cond else None,
        "model": MODEL_FILES["soft_moe"],
    })
    return res


STYLE_NAMES = ["Style 1", "Style 2", "Style 3"]


@app.post("/api/sketch")
async def sketch(file: UploadFile = File(...), style: str = Form("1")):
    t = time.perf_counter()
    img, orig = await _read(file)
    pre_ms = (time.perf_counter() - t) * 1000
    if style == "all":
        styles = [0, 1, 2]
    elif style in ("1", "2", "3"):
        styles = [int(style) - 1]
    else:
        _bad("style must be 1, 2, 3 or 'all'.")
    photo = np.repeat(to_tensor(img), len(styles), axis=0)
    (sk,), ms = run_model("face2sketch", {"photo": photo, "style": np.array(styles, dtype=np.int64)})
    return {
        "input_image": to_png_b64(img),
        "sketches": [{"style": s + 1, "name": STYLE_NAMES[s], "image": to_png_b64(from_tensor(sk[i]))}
                     for i, s in enumerate(styles)],
        "original_size": list(orig),
        "processed_size": [128, 128],
        "model": MODEL_FILES["face2sketch"],
        "timing": {"preprocess_ms": round(pre_ms, 1), "inference_ms": round(ms, 1),
                   "total_ms": round((time.perf_counter() - t) * 1000, 1)},
    }
