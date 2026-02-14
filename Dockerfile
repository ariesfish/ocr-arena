FROM node:20-alpine AS frontend-builder

WORKDIR /frontend

COPY web/package.json web/package-lock.json /frontend/
RUN npm ci

COPY web /frontend
RUN npm run build

FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OCR_ARENA_CONFIG=/app/ocr_arena/config.docker.yaml \
    VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /app

COPY .python-version /app/.python-version

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libxext6 \
    libxrender1 \
    libsm6 \
    libxcb1 \
    python3 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

RUN uv venv /opt/venv

COPY pyproject.toml uv.lock README.md /app/
COPY ocr_arena /app/ocr_arena
COPY models/pp-doclayout-v3 /opt/pp-doclayout-v3
COPY --from=frontend-builder /frontend/dist /app/frontend_dist

RUN uv sync --frozen

EXPOSE 8000

CMD ["ocr-arena-web", "--port", "8000"]
