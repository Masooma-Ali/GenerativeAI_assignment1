# GenerativeAI_assignment1

## Web application (local, without Docker)

The app has a **FastAPI** backend (`backend/`) that serves the ONNX models and a **React + Tailwind**
frontend (`frontend/`) with four workspaces: Universal Restoration, Hard-Routed Restoration,
Soft Mixture-of-Experts Restoration and Face-to-Sketch Generator, plus a Health & models page.

### 1. Model files

The backend loads these ONNX files from `src/models/` (override with the `MODEL_DIR` env var):

| Workspace | Files (produced by `scripts/export_onnx_task*.py`) |
|---|---|
| Universal Restoration | `task1_universal.onnx` |
| Hard-Routed Restoration | `task2_classifier.onnx`, `task2_specialist_{salt_pepper,blur,occlusion}.onnx` |
| Soft Mixture-of-Experts | `task3_soft_moe.onnx` |
| Face-to-Sketch Generator | `task4_face2sketch.onnx` |

Copy the `*.report.json` / `task2_onnx_report.json` files next to them to show the ONNX-vs-PyTorch
check on the Health page. A missing model only disables its workspace (HTTP 503); the rest keep working.

### 2. Backend (from the repository root)

```bash
pip install -r backend/requirements.txt
uvicorn backend.app.main:app --port 8000
```

API docs: http://localhost:8000/docs

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | server, ONNX Runtime and per-model status |
| GET | `/api/samples` | clean sample images from `backend/samples/` |
| POST | `/api/corrupt` | preview a runtime corruption |
| POST | `/api/restore/universal` | Task 1 |
| POST | `/api/restore/hard` | Task 2 (`mode` = `predicted` or `oracle`) |
| POST | `/api/restore/soft` | Task 3 |
| POST | `/api/sketch` | Task 4 (`style` = `1`, `2`, `3` or `all`) |

Restoration endpoints take a multipart `file` plus optional corruption fields:
`corruption` (`none` | `clean` | `salt_pepper` | `blur` | `occlusion`), `severity`
(`low` | `medium` | `high` = the fixed test levels, `random` = training range, `custom`), `seed`, and for
custom severity `p`, `kernel`/`sigma` or `n_rects`/`coverage`. Corruptions come from `src/data/corruptions.py`,
the same code used in training. With `corruption=none` the upload is treated as already corrupted, so no PSNR/SSIM
is reported.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. The Vite dev server proxies `/api` to `http://127.0.0.1:8000`
(set `BACKEND_URL` to change it).
