#!/usr/bin/env bash
# 在已有 lite-gpu 镜像上增量构建 ONNX 版（国内 apt/pip/conda/docker 镜像见 .env）
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/lib/load-build-env.sh
load_build_env "$(pwd)"

if ! docker image inspect visual-dps-inference-lite-gpu:latest >/dev/null 2>&1; then
  echo "缺少基底镜像 visual-dps-inference-lite-gpu:latest，请先: ./scripts/build-inference-lite-gpu-image.sh"
  exit 1
fi

echo "增量构建 visual-dps-inference-lite-gpu-onnx:latest"
echo "  APT_MIRROR=${APT_MIRROR}"
echo "  PIP_INDEX=${PIP_INDEX}"
echo "  GITHUB_PROXY_BASE=${GITHUB_PROXY_BASE:-<直连>}"
echo "  HTTP_PROXY=${HTTP_PROXY:-<无>}"
docker compose --profile inference-lite build visual-dps-inference-lite-gpu-onnx
echo "OK: visual-dps-inference-lite-gpu-onnx:latest"
