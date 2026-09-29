# Serving image: FastAPI + the model on CPU.
#
# Two stages: the builder resolves the locked dependencies with uv and then
# swaps the CUDA build of torch (several GB of NVIDIA libraries the CPU server
# never uses) for the CPU build; the runtime stage copies only the finished
# virtual environment. The image runs as an unprivileged user and keeps the
# HuggingFace cache on a volume so the 770 MB checkpoint is downloaded once.

ARG PYTHON_VERSION=3.13
ARG UV_VERSION=0.12.19
ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1

# ---------------------------------------------------------------- builder
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv
FROM python:${PYTHON_VERSION}-slim AS builder
ARG TORCH_VERSION
ARG TORCHVISION_VERSION

COPY --from=uv /uv /usr/local/bin/uv
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY convgru_ensemble/ ./convgru_ensemble/

# Locked dependencies plus the project, without the dev group.
RUN uv sync --extra serve --extra geo --no-dev --frozen

# CPU build of torch in place of the CUDA build, then drop the NVIDIA libraries
# and triton that only the CUDA build needs. Done in one layer on purpose.
RUN uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}+cpu" "torchvision==${TORCHVISION_VERSION}+cpu" \
    && extras="$(uv pip list --python .venv/bin/python --format freeze | grep -E '^(nvidia-|triton)' | cut -d= -f1)" \
    && if [ -n "$extras" ]; then uv pip uninstall --python .venv/bin/python $extras; fi \
    && find .venv -name '__pycache__' -prune -exec rm -rf {} + \
    && .venv/bin/python -c "import torch, convgru_ensemble; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# ---------------------------------------------------------------- runtime
FROM python:${PYTHON_VERSION}-slim

LABEL org.opencontainers.image.title="convgru-ensemble" \
      org.opencontainers.image.description="Ensemble precipitation nowcasting API (ConvGRU, CPU)" \
      org.opencontainers.image.source="https://github.com/Tabrigos/ConvGRU-Ensemble"

RUN groupadd --system app && useradd --system --gid app --home-dir /app --shell /usr/sbin/nologin app \
    && mkdir -p /app /data/hf && chown -R app:app /app /data

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv ./.venv
COPY --from=builder --chown=app:app /app/convgru_ensemble ./convgru_ensemble

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    DEVICE=cpu \
    HF_HOME=/data/hf \
    HF_HUB_DISABLE_TELEMETRY=1
# Model source, one of the two: a checkpoint mounted at MODEL_CHECKPOINT, or
# HF_REPO_ID (default it4lia/irene) downloaded into the HF_HOME volume.
# MAX_FORECAST_STEPS raises the cap on forecast_steps beyond the trained 12.
ENV HF_REPO_ID=it4lia/irene

VOLUME ["/data/hf"]
EXPOSE 8000
USER app

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import json,sys,urllib.request; r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)); sys.exit(0 if r.get('model_loaded') else 1)"

CMD ["uvicorn", "convgru_ensemble.serve:app", "--host", "0.0.0.0", "--port", "8000"]
