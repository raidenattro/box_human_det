#!/usr/bin/env bash
# 本地构建前端并打包 UI Docker 镜像（无需在 Docker 内拉取 node 镜像）
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "==> 构建前端 (web/dist)..."
(cd web && npm run build)

if [[ ! -f web/dist/index.html ]]; then
  echo "错误: web/dist/index.html 不存在，前端构建失败。" >&2
  exit 1
fi

echo "==> 构建 Docker 镜像 visual-dps-ui..."
docker compose --profile ui build visual-dps-ui

if [[ "${1:-}" == "--up" ]]; then
  echo "==> 启动容器..."
  docker compose --profile ui up -d visual-dps-ui
  echo "完成: http://127.0.0.1:8045"
fi
