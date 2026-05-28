#!/usr/bin/env bash
# 打最全离线包（7 镜像 + 权重 + 分卷可选）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "${ROOT}/scripts/export-offline-package.sh" --inference all --rebuild-ui "$@"
