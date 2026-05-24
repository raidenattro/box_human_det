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
docker compose build visual-dps-ui visual-dps-event-worker

if [[ "${1:-}" == "--up" ]]; then
  if [[ -z "${REDIS_PASSWORD:-}" ]]; then
    echo "错误: 请先 export REDIS_PASSWORD=... 或在本目录创建 .env（可参考 .env.example）" >&2
    exit 1
  fi
  echo "==> 启动 Redis + UI 容器..."
  docker compose up -d
  echo "完成: http://127.0.0.1:8045"
fi
