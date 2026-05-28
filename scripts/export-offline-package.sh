#!/usr/bin/env bash
# 打离线部署包：docker save 镜像 + 拷贝 app/localdata，输出目录与 .tar.gz
#
# 用法:
#   ./scripts/download-model-weights.sh          # 推荐先下载权重到 localdata
#   ./scripts/export-offline-package.sh
#   ./scripts/export-offline-package.sh --inference lite
#   ./scripts/export-offline-package.sh --inference all -o dist/my-pkg
#   ./scripts/export-offline-package.sh --no-models --no-tar
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

INFERENCE_MODE="lite"
OUTPUT=""
NO_TAR=0
INCLUDE_MODELS=1
ENV_FILE=""

usage() {
  sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
  echo ""
  echo "选项:"
  echo "  -o, --output DIR     输出目录（默认 dist/visual-dps-offline-时间戳）"
  echo "  --inference MODE     base | lite | gpu | gpu-onnx | all  （默认 lite）"
  echo "  --no-models          不打包 localdata/models/"
  echo "  --no-tar             仅生成目录，不压 tar.gz"
  echo "  --env-file PATH      打入包的 .env（默认: 项目根 .env，否则 .env.example）"
  echo "  -h, --help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output) OUTPUT="$2"; shift 2 ;;
    --inference) INFERENCE_MODE="$2"; shift 2 ;;
    --no-models) INCLUDE_MODELS=0; shift ;;
    --no-tar) NO_TAR=1; shift ;;
    --env-file) ENV_FILE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# shellcheck disable=SC1091
source "${ROOT}/scripts/lib/load-build-env.sh"
load_build_env "${ROOT}"

if [[ -z "${ENV_FILE}" ]]; then
  if [[ -f "${ROOT}/.env" ]]; then
    ENV_FILE="${ROOT}/.env"
  else
    ENV_FILE="${ROOT}/.env.example"
  fi
fi

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "错误: 找不到 env 文件: ${ENV_FILE}" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

TS="$(date +%Y%m%d-%H%M%S)"
PKG_NAME="visual-dps-offline-${TS}"
PKG_DIR="${OUTPUT:-${ROOT}/dist/${PKG_NAME}}"
mkdir -p "${PKG_DIR}/docker-images" "${PKG_DIR}/app"

require_image() {
  local img="$1"
  if ! docker image inspect "${img}" >/dev/null 2>&1; then
    echo "错误: 本地缺少镜像 ${img}（请先 build / docker pull）" >&2
    exit 1
  fi
}

maybe_image() {
  local img="$1"
  if docker image inspect "${img}" >/dev/null 2>&1; then
    IMAGES+=("${img}")
    echo "  + ${img}"
  else
    echo "  跳过（不存在）: ${img}" >&2
  fi
}

resolve_repo_tag() {
  local repo="$1"
  local preferred="${2:-}"
  if [[ -n "${preferred}" ]] && docker image inspect "${preferred}" >/dev/null 2>&1; then
    echo "${preferred}"
    return
  fi
  if docker image inspect "${repo}:latest" >/dev/null 2>&1; then
    echo "${repo}:latest"
    return
  fi
  local found
  found="$(docker images --format '{{.Repository}}:{{.Tag}}' "${repo}" 2>/dev/null | grep -v ':<none>' | grep -v ':latest' | sort | tail -n 1 || true)"
  if [[ -n "${found}" ]]; then
    echo "${found}"
    return
  fi
  echo ""
}

IMAGES=()

echo "==> 收集镜像 (inference=${INFERENCE_MODE})..."
BASE_IMAGES=(
  "redis:7"
  "bluenviron/mediamtx:1.11.3"
  "visual-dps-visual-dps-ui:latest"
  "visual-dps-event-worker:latest"
)
for img in "${BASE_IMAGES[@]}"; do
  require_image "${img}"
  IMAGES+=("${img}")
  echo "  + ${img}"
done

case "${INFERENCE_MODE}" in
  base) ;;
  lite)
    lite="$(resolve_repo_tag "visual-dps-inference-lite" "${INFERENCE_LITE_IMAGE:-}")"
    [[ -n "${lite}" ]] || { echo "错误: 缺少 visual-dps-inference-lite 镜像" >&2; exit 1; }
    require_image "${lite}"
    IMAGES+=("${lite}")
    EXPORTED_LITE_IMAGE="${lite}"
    echo "  + ${lite}"
    ;;
  gpu)
    gpu="$(resolve_repo_tag "visual-dps-inference-lite-gpu" "${INFERENCE_LITE_GPU_IMAGE:-}")"
    [[ -n "${gpu}" ]] || { echo "错误: 缺少 visual-dps-inference-lite-gpu 镜像" >&2; exit 1; }
    require_image "${gpu}"
    IMAGES+=("${gpu}")
    EXPORTED_GPU_IMAGE="${gpu}"
    echo "  + ${gpu}"
    ;;
  gpu-onnx)
    onnx="$(resolve_repo_tag "visual-dps-inference-lite-gpu-onnx" "${INFERENCE_LITE_GPU_ONNX_IMAGE:-}")"
    [[ -n "${onnx}" ]] || { echo "错误: 缺少 visual-dps-inference-lite-gpu-onnx 镜像" >&2; exit 1; }
    require_image "${onnx}"
    IMAGES+=("${onnx}")
    EXPORTED_ONNX_IMAGE="${onnx}"
    echo "  + ${onnx}"
    ;;
  all)
    lite="$(resolve_repo_tag "visual-dps-inference-lite" "${INFERENCE_LITE_IMAGE:-}")"
    gpu="$(resolve_repo_tag "visual-dps-inference-lite-gpu" "${INFERENCE_LITE_GPU_IMAGE:-}")"
    onnx="$(resolve_repo_tag "visual-dps-inference-lite-gpu-onnx" "${INFERENCE_LITE_GPU_ONNX_IMAGE:-}")"
    [[ -n "${lite}" ]] || { echo "错误: 缺少 visual-dps-inference-lite，请先 ./scripts/build-inference-lite-image.sh" >&2; exit 1; }
    [[ -n "${gpu}" ]] || { echo "错误: 缺少 visual-dps-inference-lite-gpu，请先 ./scripts/build-inference-lite-gpu-image.sh" >&2; exit 1; }
    [[ -n "${onnx}" ]] || { echo "错误: 缺少 visual-dps-inference-lite-gpu-onnx，请先 ./scripts/build-inference-lite-gpu-onnx-image.sh" >&2; exit 1; }
    require_image "${lite}"
    require_image "${gpu}"
    require_image "${onnx}"
    IMAGES+=("${lite}" "${gpu}" "${onnx}")
    EXPORTED_LITE_IMAGE="${lite}"
    EXPORTED_GPU_IMAGE="${gpu}"
    EXPORTED_ONNX_IMAGE="${onnx}"
    echo "  + ${lite}"
    echo "  + ${gpu}"
    echo "  + ${onnx}"
    ;;
  *)
    echo "错误: 未知 --inference 模式: ${INFERENCE_MODE}" >&2
    exit 1
    ;;
esac

BUNDLE="${PKG_DIR}/docker-images/bundle.tar"
echo "==> docker save -> ${BUNDLE}"
docker save -o "${BUNDLE}" "${IMAGES[@]}"
echo "    $(du -h "${BUNDLE}" | awk '{print $1}')"

APP="${PKG_DIR}/app"
echo "==> 拷贝应用文件 -> ${APP}"
cp "${ROOT}/docker-compose.yml" "${APP}/"
cp "${ROOT}/app_config.json" "${APP}/"
cp "${ROOT}/version.json" "${APP}/"
cp -a "${ROOT}/deploy" "${APP}/"

if [[ -f "${ROOT}/docker-compose.override.yml" ]]; then
  cp "${ROOT}/docker-compose.override.yml" "${APP}/"
fi

cp "${ENV_FILE}" "${APP}/.env"
if [[ ! -f "${ROOT}/.env" ]] && [[ "${ENV_FILE}" == *".env.example" ]]; then
  echo "提示: 使用 .env.example 打包，目标机 install 前请改 REDIS_PASSWORD / MEDIAMTX_PUBLIC_HOST" >&2
fi

# 同步导出的推理镜像 tag 到包内 .env
patch_env() {
  local key="$1"
  local val="$2"
  [[ -n "${val}" ]] || return 0
  if grep -q "^${key}=" "${APP}/.env" 2>/dev/null; then
    sed -i "s|^${key}=.*|${key}=${val}|" "${APP}/.env"
  else
    printf '\n%s=%s\n' "${key}" "${val}" >> "${APP}/.env"
  fi
}
patch_env "INFERENCE_LITE_IMAGE" "${EXPORTED_LITE_IMAGE:-}"
patch_env "INFERENCE_LITE_GPU_IMAGE" "${EXPORTED_GPU_IMAGE:-}"
patch_env "INFERENCE_LITE_GPU_ONNX_IMAGE" "${EXPORTED_ONNX_IMAGE:-}"

mkdir -p "${APP}/localdata/json/cameras" \
  "${APP}/localdata/logs" \
  "${APP}/localdata/inference" \
  "${APP}/localdata/frames"

RSYNC_EXCLUDES=(
  --exclude 'logs/**'
  --exclude 'frames/**'
  --exclude 'inference/**'
  --exclude 'upload/**'
  --exclude 'last_frame.jpg'
  --exclude 'json/annotation_*.json'
)
if [[ "${INCLUDE_MODELS}" -eq 0 ]]; then
  RSYNC_EXCLUDES+=(--exclude 'models/**')
fi

if [[ -d "${ROOT}/localdata" ]]; then
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${RSYNC_EXCLUDES[@]}" "${ROOT}/localdata/" "${APP}/localdata/"
  else
    echo "警告: 无 rsync，仅复制部分 localdata" >&2
    for f in camera_ips.json mediamtx.yml runtime_config.json auth_config.json auth_users.json; do
      [[ -f "${ROOT}/localdata/${f}" ]] && cp "${ROOT}/localdata/${f}" "${APP}/localdata/" || true
    done
    [[ -f "${ROOT}/localdata/json/precise_boxes_new.json" ]] \
      && cp "${ROOT}/localdata/json/precise_boxes_new.json" "${APP}/localdata/json/" || true
    if [[ "${INCLUDE_MODELS}" -eq 1 ]] && [[ -d "${ROOT}/localdata/models" ]]; then
      cp -a "${ROOT}/localdata/models" "${APP}/localdata/"
    fi
  fi
fi

if [[ ! -f "${APP}/localdata/camera_ips.json" ]]; then
  cp "${ROOT}/deploy/camera_ips.example.json" "${APP}/localdata/camera_ips.json"
  echo "  使用 deploy/camera_ips.example.json 作为 camera_ips.json"
fi
if [[ ! -f "${APP}/localdata/mediamtx.yml" ]]; then
  cp "${ROOT}/deploy/mediamtx.yml.template" "${APP}/localdata/mediamtx.yml"
fi
if [[ ! -f "${APP}/localdata/json/precise_boxes_new.json" ]]; then
  echo '{}' > "${APP}/localdata/json/precise_boxes_new.json"
fi

MODEL_CHECK_NOTE="models: ok"
if [[ "${INCLUDE_MODELS}" -eq 1 ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/deploy/check-model-weights.sh"
  echo "==> 检查推理权重..."
  if visual_dps_check_model_weights "${ROOT}/localdata" 2>/dev/null; then
    echo "    源机 localdata 权重已完整（仅 rsync 拷贝，不联网重复下载）"
  elif ! visual_dps_check_model_weights "${APP}/localdata"; then
    if command -v wget >/dev/null 2>&1 || command -v curl >/dev/null 2>&1; then
      echo "    源机/包内均不完整，仅补缺失项到离线包（已有且完整的不重下）..."
      mkdir -p "${APP}/localdata/models/rtmpose_onnx" "${APP}/localdata/models/yolo_pose"
      export DOWNLOAD_HTTP_PROXY="${BUILD_HTTP_PROXY:-${HTTP_PROXY:-}}"
      "${ROOT}/scripts/download-rtmpose-onnx-weights.sh" "${APP}/localdata/models/rtmpose_onnx"
      "${ROOT}/scripts/download-yolo-pose-weights.sh" "${APP}/localdata/models/yolo_pose"
    else
      echo "错误: 无 wget/curl，无法补全权重" >&2
      exit 1
    fi
  else
    echo "    包内已完整（由 rsync 带入，未触发下载）"
  fi
  if visual_dps_check_model_weights "${APP}/localdata"; then
    echo "    权重齐全且通过完整性校验"
  else
    MODEL_CHECK_NOTE="models: INCOMPLETE"
    echo "错误: 离线包权重缺失或不完整（仅有空/半截文件不算就绪）。" >&2
    echo "      请在本机执行: ./scripts/download-model-weights.sh 后重新 export" >&2
    exit 1
  fi
else
  MODEL_CHECK_NOTE="models: excluded (--no-models)"
fi

cp "${ROOT}/deploy/install.sh" "${PKG_DIR}/install.sh"
chmod +x "${PKG_DIR}/install.sh" "${APP}/deploy/install.sh" 2>/dev/null || true

GIT_HEAD="nogit"
if git -C "${ROOT}" rev-parse HEAD >/dev/null 2>&1; then
  GIT_HEAD="$(git -C "${ROOT}" rev-parse HEAD)"
fi

PKG_BASENAME="$(basename "${PKG_DIR}")"
INFO="${PKG_DIR}/PACKAGE_INFO.txt"
{
  echo "visual-dps offline package"
  echo "created: $(date -Iseconds)"
  echo "git: ${GIT_HEAD}"
  echo "inference_mode: ${INFERENCE_MODE}"
  echo "include_models: ${INCLUDE_MODELS}"
  echo "${MODEL_CHECK_NOTE}"
  echo "env_source: ${ENV_FILE}"
  echo ""
  echo "images:"
  printf '  %s\n' "${IMAGES[@]}"
  echo ""
  echo "install:"
  echo "  tar xzf ${PKG_BASENAME}.tar.gz"
  echo "  cd ${PKG_BASENAME} && ./install.sh"
} > "${INFO}"

TAR_PATH=""
if [[ "${NO_TAR}" -eq 0 ]]; then
  TAR_PATH="${PKG_DIR}.tar.gz"
  echo "==> 压缩 ${TAR_PATH}"
  tar -czf "${TAR_PATH}" -C "$(dirname "${PKG_DIR}")" "$(basename "${PKG_DIR}")"
fi

echo ""
echo "完成."
echo "  目录: ${PKG_DIR}"
[[ -n "${TAR_PATH}" ]] && echo "  压缩包: ${TAR_PATH}"
echo "  目标机: tar xzf ${PKG_BASENAME}.tar.gz && cd ${PKG_BASENAME} && ./install.sh"
