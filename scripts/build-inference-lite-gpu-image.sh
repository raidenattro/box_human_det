#!/usr/bin/env bash
# 构建 GPU 推理镜像（apt/pip/github 国内源 + 可选 BUILD_HTTP_PROXY）
set -euo pipefail
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
source scripts/lib/load-build-env.sh
load_build_env "$(pwd)"

echo "构建 visual-dps-inference-lite-gpu:latest"
echo "  APT_MIRROR=${APT_MIRROR}"
echo "  PIP_INDEX=${PIP_INDEX}"
echo "  GITHUB_PROXY_BASE=${GITHUB_PROXY_BASE:-<直连>}"
echo "  HTTP_PROXY=${HTTP_PROXY:-<无>}"
docker compose --profile inference-lite build visual-dps-inference-lite-gpu
echo "OK: visual-dps-inference-lite-gpu:latest（推理容器需 --gpus all）"
