"""Export the Task 4 generator to ONNX (the discriminator is a training-only component).

    python scripts/export_onnx_task4.py --ckpt gan_final/best.pt --fs2k fs2k --out onnx/task4_face2sketch.onnx
Inputs : "photo" float32 [N,3,128,128] RGB in [0,1];  "style" int64 [N] with values 0,1,2
Output : "sketch" float32 [N,1,128,128] in [0,1] (1 = white paper)"""
import argparse, json, os, sys, time
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.fs2k import FS2KDataset
from src.train_gan import load_generator

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True); ap.add_argument("--fs2k", required=True)
ap.add_argument("--out", default="onnx/task4_face2sketch.onnx"); ap.add_argument("--opset", type=int, default=17)
ap.add_argument("--tol", type=float, default=1e-4)
a = ap.parse_args(); os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
import onnx, onnxruntime as ort


class Wrapper(nn.Module):
    """[0,1] photo in, [0,1] sketch out; the [-1,1] scaling lives inside the exported graph."""
    def __init__(self, g): super().__init__(); self.g = g
    def forward(self, photo, style):
        return (self.g(photo * 2 - 1, style) + 1) / 2


G, _ = load_generator(a.ckpt, "cpu"); model = Wrapper(G).eval()
kw = dict(input_names=["photo", "style"], output_names=["sketch"], opset_version=a.opset,
          dynamic_axes={"photo": {0: "batch"}, "style": {0: "batch"}, "sketch": {0: "batch"}})
dummy = (torch.rand(1, 3, 128, 128), torch.tensor([0]))
try:
    torch.onnx.export(model, dummy, a.out, dynamo=False, **kw)
except TypeError:
    torch.onnx.export(model, dummy, a.out, **kw)
onnx.checker.check_model(onnx.load(a.out))
sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])

ds = FS2KDataset(os.path.join(a.fs2k, "val.npz"))
photos = torch.stack([(ds[i][0] + 1) / 2 for i in range(24)])
max_d = 0.0
for bs in (1, 8, 24):
    for st in (0, 1, 2):
        sty = torch.full((bs,), st, dtype=torch.long)
        with torch.no_grad(): ref = model(photos[:bs], sty).numpy()
        out = sess.run(None, {"photo": photos[:bs].numpy(), "style": sty.numpy()})[0]
        max_d = max(max_d, float(np.abs(ref - out).max()))
    print(f"batch {bs:3d}: running max|diff| = {max_d:.2e}")
one, s1 = photos[:1].numpy(), np.array([1], dtype=np.int64); sess.run(None, {"photo": one, "style": s1}); t = time.perf_counter()
for _ in range(20): sess.run(None, {"photo": one, "style": s1})
rep = {"onnx_file": os.path.basename(a.out), "size_mb": round(os.path.getsize(a.out) / 1e6, 1), "max_abs_diff": max_d,
       "tolerance": a.tol, "consistent": max_d < a.tol, "cpu_ms_batch1": round((time.perf_counter() - t) / 20 * 1000, 1), "opset": a.opset}
json.dump(rep, open(a.out + ".report.json", "w"), indent=2); print(json.dumps(rep, indent=2))
if not rep["consistent"]: sys.exit("ONNX output differs from PyTorch by more than the tolerance.")
