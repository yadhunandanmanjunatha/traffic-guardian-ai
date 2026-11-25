# Traffic Guardian AI - Docker image
# Build:  docker build -t traffic-guardian-ai .
# Run:    docker run -p 8000:8000 --env-file .env traffic-guardian-ai
# Then open http://localhost:8000

FROM python:3.11-slim

WORKDIR /app

# System deps needed by scikit-learn / shap / numpy wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching - only reinstalls
# when requirements.txt actually changes, not on every code edit)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Make sure runtime dirs exist even if .gitkeep files were excluded
RUN mkdir -p logs uploads outputs/models outputs/reports

EXPOSE 8000

# .env is NOT baked into the image (see .dockerignore) - pass it at
# `docker run` time with --env-file .env, or use environment variables
# directly (e.g. docker run -e TOMTOM_API_KEY=... )
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
