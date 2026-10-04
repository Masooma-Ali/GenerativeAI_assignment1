# Build context = repository root (the backend reuses src/data/corruptions.py from training).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/models

WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# only the corruption code is needed from src/ (no torch at inference time)
COPY src/__init__.py src/__init__.py
COPY src/data/__init__.py src/data/__init__.py
COPY src/data/corruptions.py src/data/corruptions.py
COPY backend/app backend/app
COPY backend/samples backend/samples

# ONNX models are mounted at /app/models by docker-compose (see docker-compose.yml)
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
