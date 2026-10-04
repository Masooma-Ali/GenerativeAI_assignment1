"""Lazy ONNX model registry. Missing model files do not crash the server; the affected
endpoint returns 503 and /api/health reports which files are absent."""
import json
import os
import threading
import time

import numpy as np
import onnxruntime as ort

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(REPO_ROOT, "src", "models"))

MODEL_FILES = {
    "universal": "task1_universal.onnx",
    "classifier": "task2_classifier.onnx",
    "salt_pepper": "task2_specialist_salt_pepper.onnx",
    "blur": "task2_specialist_blur.onnx",
    "occlusion": "task2_specialist_occlusion.onnx",
    "soft_moe": "task3_soft_moe.onnx",
    "face2sketch": "task4_face2sketch.onnx",
}

TASK_MODELS = {
    "universal": ["universal"],
    "hard_routing": ["classifier", "salt_pepper", "blur", "occlusion"],
    "soft_moe": ["soft_moe"],
    "face2sketch": ["face2sketch"],
}


class ModelUnavailable(RuntimeError):
    pass


class Registry:
    def __init__(self, model_dir=MODEL_DIR):
        self.model_dir = model_dir
        self._sessions = {}
        self._load_ms = {}
        self._lock = threading.Lock()

    def path(self, key):
        return os.path.join(self.model_dir, MODEL_FILES[key])

    def available(self, key):
        return os.path.isfile(self.path(key))

    def get(self, key):
        if key in self._sessions:
            return self._sessions[key]
        with self._lock:
            if key not in self._sessions:
                if not self.available(key):
                    raise ModelUnavailable(f"Model file '{MODEL_FILES[key]}' not found in {self.model_dir}.")
                t = time.perf_counter()
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._sessions[key] = ort.InferenceSession(self.path(key), opts, providers=["CPUExecutionProvider"])
                self._load_ms[key] = round((time.perf_counter() - t) * 1000, 1)
        return self._sessions[key]

    def run(self, key, feeds):
        """Run a model; returns (outputs list, elapsed ms)."""
        sess = self.get(key)
        t = time.perf_counter()
        out = sess.run(None, feeds)
        return out, (time.perf_counter() - t) * 1000

    def warmup(self):
        for key in MODEL_FILES:
            if self.available(key):
                try:
                    sess = self.get(key)
                    feeds = {i.name: (np.zeros((1,), np.int64) if "int" in i.type
                                      else np.zeros((1, 3, 128, 128), np.float32)) for i in sess.get_inputs()}
                    sess.run(None, feeds)
                except Exception as e:                       # report, don't crash startup
                    print(f"[warmup] {key}: {e}")

    def info(self, key):
        p = self.path(key)
        d = {"file": MODEL_FILES[key], "available": os.path.isfile(p), "loaded": key in self._sessions}
        if d["available"]:
            d["size_mb"] = round(os.path.getsize(p) / 1e6, 1)
        if key in self._load_ms:
            d["load_ms"] = self._load_ms[key]
        if key in self._sessions:
            s = self._sessions[key]
            d["inputs"] = [{"name": i.name, "shape": i.shape, "type": i.type} for i in s.get_inputs()]
            d["outputs"] = [{"name": o.name, "shape": o.shape, "type": o.type} for o in s.get_outputs()]
        rep = p + ".report.json"
        if os.path.isfile(rep):
            try:
                d["export_report"] = json.load(open(rep))
            except ValueError:
                pass
        return d


registry = Registry()
