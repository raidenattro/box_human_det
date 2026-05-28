#!/usr/bin/env bash
# 离线包解压后，在包根目录执行: ./install.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SCRIPT_DIR}/docker-images/bundle.tar" ]]; then
  PKG_ROOT="${SCRIPT_DIR}"
elif [[ -f "${SCRIPT_DIR}/../docker-images/bundle.tar" ]]; then
  PKG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
else
  echo "错误: 请在离线包根目录执行 ./install.sh（需含 docker-images/bundle.tar）" >&2
  exit 1
fi
BUNDLE="${PKG_ROOT}/docker-images/bundle.tar"
APP_DIR="${PKG_ROOT}/app"

if [[ ! -f "${BUNDLE}" ]]; then
  echo "错误: 未找到 ${BUNDLE}" >&2
  exit 1
fi
if [[ ! -f "${APP_DIR}/docker-compose.yml" ]]; then
  echo "错误: 未找到 ${APP_DIR}/docker-compose.yml" >&2
  exit 1
fi

echo "==> 加载 Docker 镜像..."
docker load -i "${BUNDLE}"

if [[ ! -f "${APP_DIR}/.env" ]]; then
  if [[ -f "${APP_DIR}/.env.example" ]]; then
    cp "${APP_DIR}/.env.example" "${APP_DIR}/.env"
    echo "已生成 ${APP_DIR}/.env（请编辑 REDIS_PASSWORD、MEDIAMTX_PUBLIC_HOST 后重新执行本脚本）"
    exit 1
  fi
  echo "错误: 缺少 ${APP_DIR}/.env，请参考 deploy/153.env.example 创建" >&2
  exit 1
fi

if grep -q '^REDIS_PASSWORD=change-me' "${APP_DIR}/.env" 2>/dev/null \
  || grep -q '^REDIS_PASSWORD=$' "${APP_DIR}/.env" 2>/dev/null; then
  echo "请先在 ${APP_DIR}/.env 中设置 REDIS_PASSWORD（勿使用 change-me）" >&2
  exit 1
fi

if [[ -d "${APP_DIR}/localdata/models" ]]; then
  # shellcheck disable=SC1091
  source "${APP_DIR}/deploy/check-model-weights.sh"
  echo "==> 检查推理权重完整性（离线环境必需：存在且体积达标）..."
  if ! visual_dps_check_model_weights "${APP_DIR}/localdata"; then
    echo "错误: 模型权重缺失或不完整，离线环境无法自动修复。" >&2
    echo "请换用通过 export 体积校验的离线包，或设置 VISUAL_DPS_CHECK_MODELS=0 强行启动（不推荐）。" >&2
    exit 1
  fi
elif [[ "${VISUAL_DPS_CHECK_MODELS:-1}" != "0" ]]; then
  echo "警告: 未找到 ${APP_DIR}/localdata/models/，开启检测前需自行放置权重。" >&2
fi

echo "==> 启动服务 (docker compose up -d)..."
cd "${APP_DIR}"
docker compose up -d

# shellcheck disable=SC1091
set -a && source ./.env && set +a
HOST="${MEDIAMTX_PUBLIC_HOST:-127.0.0.1}"
PORT="${UI_PORT:-8045}"
echo ""
echo "完成。浏览器访问: http://${HOST}:${PORT}/"
echo "版本检查: curl -s http://${HOST}:${PORT}/api/version"
