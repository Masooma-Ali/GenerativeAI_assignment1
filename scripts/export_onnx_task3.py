"""Export the COMPLETE soft-MoE inference pipeline (gate + 3 experts + identity branch + weighted sum)
to a single ONNX file and verify it against PyTorch.

    python scripts/export_onnx_task3.py --moe-ckpt moe_final/best.pt --data-root data --out onnx/task3_soft_moe.onnx
Input "input" [N,3,128,128] in [0,1] -> outputs "restored" [N,3,128,128] and "weights" [N,4] = [identity, salt, blur, occlusion]."""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.pets import PetRestorationDataset, load_manifest
from src.train_moe import load_moe

ap = argparse.ArgumentParser()
ap.add_argument("--moe-ckpt", required=True); ap.add_argument("--data-root", default="data")
ap.add_argument("--out", default="onnx/task3_soft_moe.onnx"); ap.add_argument("--opset", type=int, default=17)
ap.add_argument("--tol", type=float, default=1e-4)
a = ap.parse_args(); os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
import onnx, onnxruntime as ort


class Wrapper(nn.Module):
    def __init__(self, moe): super().__init__(); self.moe = moe
    def forward(self, x):
        recon, w, _ = self.moe(x); return recon, w


moe, _ = load_moe(a.moe_ckpt, "cpu"); model = Wrapper(moe).eval()
kw = dict(input_names=["input"], output_names=["restored", "weights"], opset_version=a.opset,
          dynamic_axes={"input": {0: "batch"}, "restored": {0: "batch"}, "weights": {0: "batch"}})
dummy = torch.rand(1, 3, 128, 128)
try:
    torch.onnx.export(model, dummy, a.out, dynamo=False, **kw)
except TypeError:
    torch.onnx.export(model, dummy, a.out, **kw)
onnx.checker.check_model(onnx.load(a.out))
sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])

proc = os.path.join(a.data_root, "processed")
images = np.load(os.path.join(proc, "trainval_128.npy"))
ds = PetRestorationDataset(images, manifest=load_manifest(os.path.join(proc, "val_manifest.jsonl")))
x = torch.stack([ds[i][0] for i in range(64)])
dr = dw = 0.0
for bs in (1, 8, 64):
    with torch.no_grad(): ref_r, ref_w = model(x[:bs])
    o_r, o_w = sess.run(None, {"input": x[:bs].numpy()})
    dr = max(dr, float(np.abs(ref_r.numpy() - o_r).max())); dw = max(dw, float(np.abs(ref_w.numpy() - o_w).max()))
    print(f"batch {bs:3d}: max|diff| restored {np.abs(ref_r.numpy() - o_r).max():.2e}  weights {np.abs(ref_w.numpy() - o_w).max():.2e}")
one = x[:1].numpy(); sess.run(None, {"input": one}); t = time.perf_counter()
for _ in range(20): sess.run(None, {"input": one})
rep = {"onnx_file": os.path.basename(a.out), "size_mb": round(os.path.getsize(a.out) / 1e6, 1),
       "max_abs_diff_restored": dr, "max_abs_diff_weights": dw, "tolerance": a.tol,
       "consistent": max(dr, dw) < a.tol, "cpu_ms_batch1": round((time.perf_counter() - t) / 20 * 1000, 1), "opset": a.opset}
json.dump(rep, open(a.out + ".report.json", "w"), indent=2); print(json.dumps(rep, indent=2))
if not rep["consistent"]: sys.exit("ONNX output differs from PyTorch by more than the tolerance.")
