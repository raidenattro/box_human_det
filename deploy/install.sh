#!/usr/bin/env bash
# 离线包根目录: ./install.sh [--host IP] [--weights-dir DIR] [--stop-infer]
set -euo pipefail

HOST_IP=""
WEIGHTS_DIR=""
STOP_INFER=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST_IP="$2"; shift 2 ;;
    --weights-dir) WEIGHTS_DIR="$2"; shift 2 ;;
    --stop-infer) STOP_INFER=1; shift ;;
    -h|--help)
      cat <<'EOF'
用法: ./install.sh [--host IP] [--weights-dir DIR] [--stop-infer]

  --weights-dir  默认 <包根>/weights；兼容旧包 app/localdata/models
EOF
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
  echo "错误: 请在离线包根目录执行（需含 docker-images/bundle.tar）" >&2
  exit 1
fi

BUNDLE="${PKG_ROOT}/docker-images/bundle.tar"
APP_DIR="${PKG_ROOT}/app"
COMPOSE_FILE="${APP_DIR}/docker-compose.deploy.yml"
[[ -f "${COMPOSE_FILE}" ]] || COMPOSE_FILE="${APP_DIR}/docker-compose.yml"

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
  echo "错误: 未找到 bundle.tar" >&2
  exit 1
fi

if [[ ! -f "${APP_DIR}/.env" ]]; then
  if [[ -f "${APP_DIR}/deploy/offline.env.example" ]]; then
    cp "${APP_DIR}/deploy/offline.env.example" "${APP_DIR}/.env"
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
  echo "请先在 ${APP_DIR}/.env 中设置 REDIS_PASSWORD" >&2
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

install_weights() {
  local src="$1"
  local dest="${APP_DIR}/localdata/models"
  mkdir -p "${dest}/rtmpose_onnx" "${dest}/yolo_pose"
  echo "==> 安装权重 ${src} -> ${dest}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${src}/rtmpose_onnx/" "${dest}/rtmpose_onnx/"
    rsync -a "${src}/yolo_pose/" "${dest}/yolo_pose/"
  else
    cp -a "${src}/rtmpose_onnx/." "${dest}/rtmpose_onnx/"
    cp -a "${src}/yolo_pose/." "${dest}/yolo_pose/"
  fi
}

# shellcheck disable=SC1091
source "${APP_DIR}/deploy/check-model-weights.sh"

if [[ -z "${WEIGHTS_DIR}" ]]; then
  if [[ -d "${PKG_ROOT}/weights" ]]; then
    WEIGHTS_DIR="${PKG_ROOT}/weights"
  elif [[ -d "${APP_DIR}/localdata/models" ]]; then
    WEIGHTS_DIR="${APP_DIR}/localdata/models"
  fi
fi

if [[ -n "${WEIGHTS_DIR}" && -d "${WEIGHTS_DIR}" ]]; then
  if [[ "${WEIGHTS_DIR}" != "${APP_DIR}/localdata/models" ]]; then
    install_weights "${WEIGHTS_DIR}"
  fi
  echo "==> 检查推理权重..."
  visual_dps_check_model_weights "${APP_DIR}/localdata" || {
    echo "错误: 模型权重缺失或不完整" >&2
    exit 1
  }
  if [[ -f "${WEIGHTS_DIR}/SHA256SUMS" ]] && [[ "${WEIGHTS_DIR}" == "${PKG_ROOT}/weights" ]]; then
    echo "==> 校验 SHA256SUMS..."
  (cd "${WEIGHTS_DIR}" && sha256sum -c SHA256SUMS)
  fi
else
  echo "警告: 未找到 weights/，且 app 内无 models（推理将不可用）" >&2
fi

echo "==> 停止旧 compose 栈..."
cd "${APP_DIR}"
compose_cmd down 2>/dev/null || true

if [[ "${STOP_INFER}" -eq 1 ]]; then
  docker ps -a --format '{{.Names}}' | grep -E '^visual-dps-infer-' | xargs -r docker rm -f || true
fi

echo "==> 启动服务..."
compose_cmd up -d

set -a && source ./.env && set +a
HOST="${MEDIAMTX_PUBLIC_HOST:-127.0.0.1}"
PORT="${UI_PORT:-8045}"
sleep 3
echo ""
echo "完成。http://${HOST}:${PORT}/"
echo "版本: curl -s http://${HOST}:${PORT}/api/version"
