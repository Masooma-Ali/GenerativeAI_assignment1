
# RestoreLab

**Image restoration with autoencoders and mixtures of experts, plus style-conditioned face-to-sketch generation, served through one web app.**

RestoreLab is a Generative AI course project (Assignment 1). It builds four related systems, trains and tunes them with Optuna, tracks the experiments in MLflow, exports the inference models to ONNX, and serves them from a FastAPI + React application.

## The four tasks

| # | Task | Idea |
|---|------|------|
| 1 | **Universal restoration** | One denoising autoencoder with a convolutional bottleneck restores clean, salt-and-pepper, blurred and occluded pet images (L1 + SSIM loss). |
| 2 | **Hard-routed restoration** | A corruption classifier picks one of three specialist autoencoders; clean images take an identity bypass. Evaluated with oracle and predicted routing. |
| 3 | **Soft mixture-of-experts** | A gate (initialised from the classifier) blends the three experts and an identity branch; gate warm-up, joint fine-tuning, load-balance loss, collapse detection. |
| 4 | **Face-to-sketch cGAN** | U-Net generator + PatchGAN discriminator conditioned on one of three sketch styles (embedding + FiLM), trained on FS2K. |

Datasets: [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/) (Tasks 1-3, 128x128, seed-42 80/20 split, runtime corruptions with fixed validation/test manifests) and FS2K (Task 4, official split, stratified 15% validation).

## Highlights

* A fully connected bottleneck plateaued at ~18 dB PSNR whatever its size; a convolutional 8x8x64 bottleneck reached **25.8 dB** (validation).
* Test set: universal model gains **+10.2 dB** on salt-and-pepper and **+8.5 dB** on occlusion; it does not help on blur.
* Corruption classifier: **99.78%** test accuracy (80 errors in 36,690 samples).
* Soft MoE ~ hard routing on single corruptions; on mixed corruptions the universal model wins (analysed in the report).
* cGAN: validation SSIM **0.524** vs 0.288 for a grayscale-photo baseline; a style-swap test confirms the style embedding is used.
* All inference models exported to ONNX and verified against PyTorch (max difference < 3e-6).

## Tech stack

PyTorch, Optuna, MLflow (SQLite backend), ONNX / ONNX Runtime, FastAPI, React + Tailwind CSS (Vite), Docker Compose.

## Repository layout

```
src/            data pipeline, models, losses, training loops
scripts/        data preparation, Optuna studies, training, evaluation, ONNX export
tests/          unit tests for the corruption pipeline
app/backend/    FastAPI service (ONNX Runtime) + API tests
app/frontend/   React + Tailwind application
report/         IEEE-format LaTeX report
```

## Links
* Demo video: https://youtu.be/zQ54KmZHTcA

## Quick start (Docker)

1. Put the ONNX model files in `src/models/` (see the table below).
2. From the repository root run:

```bash
docker compose up --build
```

3. Open http://localhost:8080. The API and its Swagger UI are also at http://localhost:8000/docs.

`src/models/` and `backend/samples/` are mounted into the backend container, so adding a model
or sample image only needs `docker compose restart backend`, not a rebuild. Stop with `docker compose down`.

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
