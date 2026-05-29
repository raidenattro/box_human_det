#!/usr/bin/env bash
# 全量通用离线包：7 镜像 + weights/，默认目录输出（不 gzip 整包）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "${ROOT}/scripts/export-offline-package.sh" --inference all "$@"
