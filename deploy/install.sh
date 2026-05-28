#!/usr/bin/env bash
# 离线包解压后，在包根目录执行: ./install.sh [--host IP] [--stop-infer]
set -euo pipefail

HOST_IP=""
STOP_INFER=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST_IP="$2"; shift 2 ;;
    --stop-infer) STOP_INFER=1; shift ;;
    -h|--help)
      echo "用法: ./install.sh [--host 192.168.0.204] [--stop-infer]"
      exit 0
      ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

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
COMPOSE_FILE="${APP_DIR}/docker-compose.deploy.yml"
if [[ ! -f "${COMPOSE_FILE}" ]]; then
  COMPOSE_FILE="${APP_DIR}/docker-compose.yml"
fi

compose_cmd() {
  if docker compose version >/dev/null 2>&1; then
    docker compose -f "${COMPOSE_FILE}" "$@"
  elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose -f "${COMPOSE_FILE}" "$@"
  else
    echo "错误: 需要 docker compose 或 docker-compose" >&2
    exit 1
  fi
}

if [[ ! -f "${BUNDLE}" ]] && ! ls "${PKG_ROOT}"/docker-images/bundle.tar.part-* >/dev/null 2>&1; then
  echo "错误: 未找到 bundle.tar 或 bundle.tar.part-*" >&2
  exit 1
fi

if [[ ! -f "${APP_DIR}/.env" ]]; then
  if [[ -f "${APP_DIR}/deploy/offline.env.example" ]]; then
    cp "${APP_DIR}/deploy/offline.env.example" "${APP_DIR}/.env"
    echo "已从 deploy/offline.env.example 生成 .env"
  elif [[ -f "${APP_DIR}/.env.example" ]]; then
    cp "${APP_DIR}/.env.example" "${APP_DIR}/.env"
  else
    echo "错误: 缺少 ${APP_DIR}/.env" >&2
    exit 1
  fi
fi

if [[ -n "${HOST_IP}" ]]; then
  if grep -q '^MEDIAMTX_PUBLIC_HOST=' "${APP_DIR}/.env"; then
    sed -i "s|^MEDIAMTX_PUBLIC_HOST=.*|MEDIAMTX_PUBLIC_HOST=${HOST_IP}|" "${APP_DIR}/.env"
  else
    echo "MEDIAMTX_PUBLIC_HOST=${HOST_IP}" >> "${APP_DIR}/.env"
  fi
  echo "已设置 MEDIAMTX_PUBLIC_HOST=${HOST_IP}"
fi

grep -q '^VISUAL_DPS_IMAGE_TAG=' "${APP_DIR}/.env" 2>/dev/null \
  || echo 'VISUAL_DPS_IMAGE_TAG=latest' >> "${APP_DIR}/.env"

if grep -q '^REDIS_PASSWORD=change-me' "${APP_DIR}/.env" 2>/dev/null \
  || grep -q '^REDIS_PASSWORD=$' "${APP_DIR}/.env" 2>/dev/null; then
  echo "请先在 ${APP_DIR}/.env 中设置 REDIS_PASSWORD（勿使用 change-me）" >&2
  exit 1
fi

echo "==> 加载 Docker 镜像..."
if [[ -f "${BUNDLE}" ]]; then
  docker load -i "${BUNDLE}"
else
  cat "${PKG_ROOT}"/docker-images/bundle.tar.part-* > "${PKG_ROOT}/docker-images/bundle.tar.assembled"
  docker load -i "${PKG_ROOT}/docker-images/bundle.tar.assembled"
  rm -f "${PKG_ROOT}/docker-images/bundle.tar.assembled"
fi

if [[ -d "${APP_DIR}/localdata/models" ]]; then
  # shellcheck disable=SC1091
  source "${APP_DIR}/deploy/check-model-weights.sh"
  echo "==> 检查推理权重完整性..."
  if ! visual_dps_check_model_weights "${APP_DIR}/localdata"; then
    echo "错误: 模型权重缺失或不完整。" >&2
    exit 1
  fi
fi

echo "==> 停止旧 compose 栈..."
cd "${APP_DIR}"
compose_cmd down 2>/dev/null || true

if [[ "${STOP_INFER}" -eq 1 ]]; then
  echo "==> 停止旧推理容器 visual-dps-infer-* ..."
  docker ps -a --format '{{.Names}}' | grep -E '^visual-dps-infer-' | xargs -r docker rm -f || true
fi

echo "==> 启动服务..."
compose_cmd up -d

# shellcheck disable=SC1091
set -a && source ./.env && set +a
HOST="${MEDIAMTX_PUBLIC_HOST:-127.0.0.1}"
PORT="${UI_PORT:-8045}"
sleep 3
echo ""
echo "完成。浏览器访问: http://${HOST}:${PORT}/"
echo "版本: curl -s http://${HOST}:${PORT}/api/version"
