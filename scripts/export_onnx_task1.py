"""Export the Task 1 autoencoder to ONNX and verify it against PyTorch.

    python scripts/export_onnx_task1.py --ckpt /kaggle/working/task1_final/best.pt \
        --data-root /kaggle/working/data --out /kaggle/working/onnx/task1_universal.onnx

Input : "input"    float32 [N,3,128,128], RGB in [0,1]
Output: "restored" float32 [N,3,128,128], RGB in [0,1]   (batch size is dynamic)
Writes <out>.report.json with max/mean absolute difference vs PyTorch and CPU latency.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.pets import PetRestorationDataset, load_manifest
from src.train_ae import build_model

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--data-root", default="data"); ap.add_argument("--opset", type=int, default=17)
ap.add_argument("--tol", type=float, default=1e-4)
a = ap.parse_args()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

ck = torch.load(a.ckpt, map_location="cpu")
model = build_model(ck["cfg"]); model.load_state_dict(ck["model"]); model.eval()   # eval(): dropout off, BN uses running stats

dummy = torch.rand(1, 3, 128, 128)
kw = dict(input_names=["input"], output_names=["restored"], opset_version=a.opset,
          dynamic_axes={"input": {0: "batch"}, "restored": {0: "batch"}})
try:
    torch.onnx.export(model, dummy, a.out, dynamo=False, **kw)      # classic exporter (torch >= 2.5)
except TypeError:
    torch.onnx.export(model, dummy, a.out, **kw)

import onnx, onnxruntime as ort
onnx.checker.check_model(onnx.load(a.out))
sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])

# realistic test inputs: corrupted validation images
proc = os.path.join(a.data_root, "processed")
images = np.load(os.path.join(proc, "trainval_128.npy"))
ds = PetRestorationDataset(images, manifest=load_manifest(os.path.join(proc, "val_manifest.jsonl")))
x = torch.stack([ds[i][0] for i in range(64)])

max_d, mean_d = 0.0, []
for bs in (1, 8, 64):                                   # also proves the batch axis is dynamic
    xb = x[:bs]
    with torch.no_grad():
        ref = model(xb).numpy()
    out = sess.run(None, {"input": xb.numpy()})[0]
    d = np.abs(ref - out)
    max_d = max(max_d, float(d.max())); mean_d.append(float(d.mean()))
    print(f"batch {bs:3d}: max|diff| = {d.max():.2e}  mean|diff| = {d.mean():.2e}")

one = x[:1].numpy(); sess.run(None, {"input": one})     # warm-up
t = time.perf_counter()
for _ in range(50): sess.run(None, {"input": one})
ms = (time.perf_counter() - t) / 50 * 1000

rep = {"onnx_file": os.path.basename(a.out), "size_mb": round(os.path.getsize(a.out) / 1e6, 1),
       "max_abs_diff": max_d, "mean_abs_diff": float(np.mean(mean_d)), "tolerance": a.tol,
       "consistent": max_d < a.tol, "cpu_latency_ms_batch1": round(ms, 1), "opset": a.opset}
json.dump(rep, open(a.out + ".report.json", "w"), indent=2)
print(json.dumps(rep, indent=2))
if not rep["consistent"]:
    sys.exit("ONNX output differs from PyTorch by more than the tolerance - investigate before using it.")
