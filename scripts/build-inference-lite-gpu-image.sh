#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose --profile inference-lite build visual-dps-inference-lite-gpu
echo "OK: visual-dps-inference-lite-gpu:latest（启动推理需 --gpus all）"
