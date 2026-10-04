"""Export the Task 2 classifier (with softmax) and the three specialists to ONNX and verify them
against PyTorch, individually AND as a complete hard-routed pipeline (routing done with onnxruntime).

    python scripts/export_onnx_task2.py --clf-ckpt clf_final/best.pt --spec-dir specialists \
        --data-root data --out-dir onnx
Files: task2_classifier.onnx (input -> probabilities), task2_specialist_{salt_pepper,blur,occlusion}.onnx"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.corruptions import CLASSES
from src.data.pets import PetRestorationDataset, load_manifest
from src.pipeline import HardRouter

ap = argparse.ArgumentParser()
ap.add_argument("--clf-ckpt", required=True); ap.add_argument("--spec-dir", required=True)
ap.add_argument("--data-root", default="data"); ap.add_argument("--out-dir", default="onnx")
ap.add_argument("--opset", type=int, default=17); ap.add_argument("--tol", type=float, default=1e-4)
a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
import onnx, onnxruntime as ort


class ClassifierProbs(nn.Module):
    def __init__(self, clf): super().__init__(); self.clf = clf
    def forward(self, x): return torch.softmax(self.clf(x), 1)


def export(model, path, out_name):
    kw = dict(input_names=["input"], output_names=[out_name], opset_version=a.opset,
              dynamic_axes={"input": {0: "batch"}, out_name: {0: "batch"}})
    dummy = torch.rand(1, 3, 128, 128)
    try:
        torch.onnx.export(model, dummy, path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, dummy, path, **kw)
    onnx.checker.check_model(onnx.load(path))
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])


router = HardRouter(a.clf_ckpt, a.spec_dir, "cpu")
proc = os.path.join(a.data_root, "processed")
images = np.load(os.path.join(proc, "trainval_128.npy"))
ds = PetRestorationDataset(images, manifest=load_manifest(os.path.join(proc, "val_manifest.jsonl")))
x = torch.stack([ds[i][0] for i in range(64)])

models = {"classifier": (ClassifierProbs(router.clf).eval(), "task2_classifier.onnx", "probabilities")}
for name in CLASSES[1:]:
    models[name] = (router.experts[name], f"task2_specialist_{name}.onnx", "restored")

report, sessions = {}, {}
for key, (m, fname, out_name) in models.items():
    path = os.path.join(a.out_dir, fname); sess = export(m, path, out_name); sessions[key] = sess
    max_d = 0.0
    for bs in (1, 8, 64):
        with torch.no_grad(): ref = m(x[:bs]).numpy()
        max_d = max(max_d, float(np.abs(ref - sess.run(None, {"input": x[:bs].numpy()})[0]).max()))
    one = x[:1].numpy(); sess.run(None, {"input": one}); t = time.perf_counter()
    for _ in range(30): sess.run(None, {"input": one})
    report[key] = {"file": fname, "size_mb": round(os.path.getsize(path) / 1e6, 1), "max_abs_diff": max_d,
                   "consistent": max_d < a.tol, "cpu_ms_batch1": round((time.perf_counter() - t) / 30 * 1000, 1)}
    print(key, report[key])

# end-to-end hard-routed pipeline with onnxruntime vs the PyTorch router
probs = sessions["classifier"].run(None, {"input": x.numpy()})[0]; route = probs.argmax(1)
out = x.numpy().copy()
for k, name in enumerate(CLASSES):
    sel = np.where(route == k)[0]
    if k > 0 and len(sel): out[sel] = np.clip(sessions[name].run(None, {"input": x.numpy()[sel]})[0], 0, 1)
ref, _, ref_route = router(x, "predicted")
report["pipeline"] = {"route_mismatches": int((ref_route.numpy() != route).sum()),
                      "max_abs_diff": float(np.abs(ref.numpy() - out).max())}
report["pipeline"]["consistent"] = report["pipeline"]["max_abs_diff"] < a.tol and report["pipeline"]["route_mismatches"] == 0
print("pipeline", report["pipeline"])
json.dump(report, open(os.path.join(a.out_dir, "task2_onnx_report.json"), "w"), indent=2)
if not all(v["consistent"] for v in report.values()):
    sys.exit("At least one ONNX model/pipeline differs from PyTorch by more than the tolerance.")
