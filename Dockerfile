FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    DATA_DIR=/data \
    NAFNET_AUTODOWNLOAD=0 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    TORCH_NUM_THREADS=2

# CPU-only PyTorch, pinned so the image is reproducible.
RUN pip install torch==2.3.1+cpu --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Hugging Face Spaces runs containers as user 1000.
RUN useradd --create-home --uid 1000 app && mkdir -p /data && chown app /data
COPY --chown=app . .
USER app

EXPOSE 7860
CMD ["python", "app.py"]
