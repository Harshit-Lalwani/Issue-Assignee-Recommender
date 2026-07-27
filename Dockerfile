# Serving container for the Issue-Assignee Recommender API (Hugging Face Spaces, Docker SDK).
# Free-tier Spaces are CPU-only -- installing CPU-only torch explicitly, before anything that
# would otherwise pull the default CUDA-enabled wheel, is what keeps this build small and fast
# (the default wheel drags in ~2GB of CUDA libraries that go unused and can blow build limits).
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# src/ and data/ stay siblings under /app, exactly like the local dev layout -- api.py resolves
# data/models and data/processed relative to its own file location, so this avoids needing any
# path environment variables just to match what running with `--app-dir src` locally does.
COPY src/ src/
COPY data/processed/phase3_results.csv data/processed/phase3_results.csv
COPY data/processed/split_manifest.csv data/processed/split_manifest.csv

# Hugging Face Spaces expects the app to listen on 7860.
EXPOSE 7860

CMD ["uvicorn", "issue_assignee_recommender.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "7860"]
