FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install openmim && \
    mim install "mmengine>=0.10.0" "mmcv>=2.0.0,<2.3.0" "mmdet>=3.0.0,<3.4.0"

RUN pip install \
    mmpose \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    opencv-python-headless \
    numpy \
    psutil

COPY . /app

EXPOSE 8045

CMD ["python", "main.py"]
