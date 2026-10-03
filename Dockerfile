# syntax=docker/dockerfile:1
# MedicalModel2024 — Cancer Screening Microsimulation (Flask web UI)
#
# Build:   docker build -t med_flask_app:latest .
# Run:     docker run --rm -p 5000:5000 --name flask_tutorial med_flask_app:latest
# Compose: docker compose up --build

FROM python:3.11-slim

# --- Runtime environment ------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    NUMBA_CACHE_DIR=/app/.numba_cache \
    FLASK_APP=web_app.py \
    HOST=0.0.0.0 \
    PORT=5000

WORKDIR /app

# --- System dependencies ------------------------------------------------------
# tini gives a proper init (signal handling); libgomp1 is required by NumPy/Numba.
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Python dependencies (separate layer for better build caching) -------------
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# --- Application code ---------------------------------------------------------
COPY . .

# Warm up the Numba JIT cache and validate the build with a tiny run.
# Tolerant: a failure (e.g. missing data files) does not break the image —
# Numba will simply compile on first use at runtime.
RUN python -c "from model.simulation import Simulation; \
s = Simulation(param_source='config/parameters.toml', seed=1); \
s.params.init_population = 1000; \
s.params.years_to_simulate = 2; \
s.run_full_simulation(verbose=False); \
print('numba warm-up OK')" \
    || echo "numba warm-up skipped (will compile on first run)"

# Run as a non-root user; /app stays writable for output/ and the numba cache.
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/output /app/.numba_cache \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# --- Health check -------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:5000/api/status', timeout=3).status == 200 else 1)"

# --- Entry point --------------------------------------------------------------
# The web app keeps run state in module-level globals, so it must run as a
# single process (no multi-worker WSGI server).
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "web_app.py", "--host", "0.0.0.0", "--port", "5000"]
