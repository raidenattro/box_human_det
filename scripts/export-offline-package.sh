#!/usr/bin/env bash
# 打离线部署包：docker save 镜像 + 拷贝 app/localdata，输出目录与 .tar.gz
#
# 用法:
#   ./scripts/download-model-weights.sh
#   ./scripts/export-offline-package.sh --inference all --rebuild-ui
#   ./scripts/export-offline-package.sh --inference all --split 2G
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

INFERENCE_MODE="lite"
OUTPUT=""
NO_TAR=0
INCLUDE_MODELS=1
ENV_FILE=""
REBUILD_UI=0
SPLIT_SIZE=""

usage() {
  sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
  echo ""
  echo "选项:"
  echo "  -o, --output DIR       输出目录（默认 dist/visual-dps-offline[-complete]-时间戳）"
  echo "  --inference MODE       base | lite | gpu | gpu-onnx | all  （完整包用 all）"
  echo "  --rebuild-ui           打包前先 ./scripts/build-ui-image.sh（UI 与 event-worker 同版）"
  echo "  --split SIZE           将最终 tar.gz 分卷（如 2G、2048M）"
  echo "  --no-models            不打包 localdata/models/"
  echo "  --no-tar               仅生成目录，不压 tar.gz"
  echo "  --env-file PATH        打入包的 .env"
  echo "  -h, --help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -o|--output) OUTPUT="$2"; shift 2 ;;
    --inference) INFERENCE_MODE="$2"; shift 2 ;;
    --rebuild-ui) REBUILD_UI=1; shift ;;
    --split) SPLIT_SIZE="$2"; shift 2 ;;
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

if [[ "${REBUILD_UI}" -eq 1 ]]; then
  echo "==> 构建 UI + Event 镜像..."
  "${ROOT}/scripts/build-ui-image.sh"
fi

if [[ -z "${ENV_FILE}" ]]; then
  if [[ -f "${ROOT}/.env" ]]; then
    ENV_FILE="${ROOT}/.env"
  else
    ENV_FILE="${ROOT}/deploy/offline.env.example"
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
if [[ "${INFERENCE_MODE}" == "all" ]]; then
  PKG_NAME="visual-dps-offline-complete-${TS}"
else
  PKG_NAME="visual-dps-offline-${TS}"
fi
PKG_DIR="${OUTPUT:-${ROOT}/dist/${PKG_NAME}}"
mkdir -p "${PKG_DIR}/docker-images" "${PKG_DIR}/app"

require_image() {
  local img="$1"
  if ! docker image inspect "${img}" >/dev/null 2>&1; then
    echo "错误: 本地缺少镜像 ${img}" >&2
    exit 1
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
  docker images --format '{{.Repository}}:{{.Tag}}' "${repo}" 2>/dev/null | grep -v ':<none>' | grep -v ':latest' | sort | tail -n 1 || true
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
    [[ -n "${lite}" ]] || { echo "错误: 缺少 visual-dps-inference-lite" >&2; exit 1; }
    require_image "${lite}"
    IMAGES+=("${lite}")
    EXPORTED_LITE_IMAGE="${lite}"
    echo "  + ${lite}"
    ;;
  gpu)
    gpu="$(resolve_repo_tag "visual-dps-inference-lite-gpu" "${INFERENCE_LITE_GPU_IMAGE:-}")"
    [[ -n "${gpu}" ]] || { echo "错误: 缺少 visual-dps-inference-lite-gpu" >&2; exit 1; }
    require_image "${gpu}"
    IMAGES+=("${gpu}")
    EXPORTED_GPU_IMAGE="${gpu}"
    echo "  + ${gpu}"
    ;;
  gpu-onnx)
    onnx="$(resolve_repo_tag "visual-dps-inference-lite-gpu-onnx" "${INFERENCE_LITE_GPU_ONNX_IMAGE:-}")"
    [[ -n "${onnx}" ]] || { echo "错误: 缺少 visual-dps-inference-lite-gpu-onnx" >&2; exit 1; }
    require_image "${onnx}"
    IMAGES+=("${onnx}")
    EXPORTED_ONNX_IMAGE="${onnx}"
    echo "  + ${onnx}"
    ;;
  all)
    lite="$(resolve_repo_tag "visual-dps-inference-lite" "${INFERENCE_LITE_IMAGE:-}")"
    gpu="$(resolve_repo_tag "visual-dps-inference-lite-gpu" "${INFERENCE_LITE_GPU_IMAGE:-}")"
    onnx="$(resolve_repo_tag "visual-dps-inference-lite-gpu-onnx" "${INFERENCE_LITE_GPU_ONNX_IMAGE:-}")"
    [[ -n "${lite}" ]] || { echo "错误: 缺少 visual-dps-inference-lite" >&2; exit 1; }
    [[ -n "${gpu}" ]] || { echo "错误: 缺少 visual-dps-inference-lite-gpu" >&2; exit 1; }
    [[ -n "${onnx}" ]] || { echo "错误: 缺少 visual-dps-inference-lite-gpu-onnx" >&2; exit 1; }
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
echo "==> 拷贝应用 -> ${APP}"
cp "${ROOT}/docker-compose.yml" "${APP}/"
cp "${ROOT}/docker-compose.deploy.yml" "${APP}/"
cp "${ROOT}/app_config.json" "${APP}/"
cp "${ROOT}/version.json" "${APP}/"
cp -a "${ROOT}/deploy" "${APP}/"
[[ -f "${ROOT}/docker-compose.override.yml" ]] && cp "${ROOT}/docker-compose.override.yml" "${APP}/"

cp "${ENV_FILE}" "${APP}/.env"
cp "${ROOT}/deploy/offline.env.example" "${APP}/deploy/" 2>/dev/null || true

patch_env() {
  local key="$1" val="$2"
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
grep -q '^VISUAL_DPS_IMAGE_TAG=' "${APP}/.env" 2>/dev/null || echo 'VISUAL_DPS_IMAGE_TAG=latest' >> "${APP}/.env"

mkdir -p "${APP}/localdata/json/cameras" "${APP}/localdata/logs" "${APP}/localdata/inference" "${APP}/localdata/frames"
RSYNC_EXCLUDES=(--exclude 'logs/**' --exclude 'frames/**' --exclude 'inference/**' --exclude 'upload/**' --exclude 'last_frame.jpg' --exclude 'json/annotation_*.json')
[[ "${INCLUDE_MODELS}" -eq 0 ]] && RSYNC_EXCLUDES+=(--exclude 'models/**')

if [[ -d "${ROOT}/localdata" ]]; then
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${RSYNC_EXCLUDES[@]}" "${ROOT}/localdata/" "${APP}/localdata/"
  elif [[ "${INCLUDE_MODELS}" -eq 1 ]] && [[ -d "${ROOT}/localdata/models" ]]; then
    cp -a "${ROOT}/localdata/models" "${APP}/localdata/" 2>/dev/null || true
  fi
fi

[[ -f "${APP}/localdata/camera_ips.json" ]] || cp "${ROOT}/deploy/camera_ips.example.json" "${APP}/localdata/camera_ips.json"
[[ -f "${APP}/localdata/mediamtx.yml" ]] || cp "${ROOT}/deploy/mediamtx.yml.template" "${APP}/localdata/mediamtx.yml"
[[ -f "${APP}/localdata/json/precise_boxes_new.json" ]] || echo '{}' > "${APP}/localdata/json/precise_boxes_new.json"

MODEL_CHECK_NOTE="models: ok"
if [[ "${INCLUDE_MODELS}" -eq 1 ]]; then
  # shellcheck disable=SC1091
  source "${ROOT}/deploy/check-model-weights.sh"
  echo "==> 检查推理权重..."
  if ! visual_dps_check_model_weights "${APP}/localdata"; then
    if command -v wget >/dev/null 2>&1 || command -v curl >/dev/null 2>&1; then
      echo "    补缺失权重到离线包..."
      mkdir -p "${APP}/localdata/models/rtmpose_onnx" "${APP}/localdata/models/yolo_pose"
      export DOWNLOAD_HTTP_PROXY="${BUILD_HTTP_PROXY:-${HTTP_PROXY:-}}"
      "${ROOT}/scripts/download-rtmpose-onnx-weights.sh" "${APP}/localdata/models/rtmpose_onnx"
      "${ROOT}/scripts/download-yolo-pose-weights.sh" "${APP}/localdata/models/yolo_pose"
    else
      echo "错误: 无 wget/curl" >&2; exit 1
    fi
  fi
  visual_dps_check_model_weights "${APP}/localdata" || { echo "错误: 权重不完整" >&2; exit 1; }
  echo "    权重齐全"
else
  MODEL_CHECK_NOTE="models: excluded"
fi

cp "${ROOT}/deploy/install.sh" "${PKG_DIR}/install.sh"
cp "${ROOT}/deploy/OFFLINE-QUICKSTART.md" "${PKG_DIR}/OFFLINE-QUICKSTART.md"
cp "${ROOT}/deploy/verify-package.sh" "${PKG_DIR}/verify-package.sh"
chmod +x "${PKG_DIR}/install.sh" "${PKG_DIR}/verify-package.sh" "${APP}/deploy/"*.sh 2>/dev/null || true

GIT_HEAD="nogit"
git -C "${ROOT}" rev-parse HEAD >/dev/null 2>&1 && GIT_HEAD="$(git -C "${ROOT}" rev-parse HEAD)"
PKG_BASENAME="$(basename "${PKG_DIR}")"

{
  echo "visual-dps offline package"
  echo "created: $(date -Iseconds)"
  echo "git: ${GIT_HEAD}"
  echo "inference_mode: ${INFERENCE_MODE}"
  echo "include_models: ${INCLUDE_MODELS}"
  echo "${MODEL_CHECK_NOTE}"
  echo "env_source: ${ENV_FILE}"
  echo "images:"
  printf '  %s\n' "${IMAGES[@]}"
  echo ""
  echo "install:"
  echo "  tar xzf ${PKG_BASENAME}.tar.gz   # 或 cat ${PKG_BASENAME}.tar.gz.part-* > ${PKG_BASENAME}.tar.gz"
  echo "  cd ${PKG_BASENAME} && ./verify-package.sh && ./install.sh --host <IP> --stop-infer"
} > "${PKG_DIR}/PACKAGE_INFO.txt"

echo "==> 包内校验..."
"${PKG_DIR}/verify-package.sh" "${PKG_DIR}"

TAR_PATH=""
if [[ "${NO_TAR}" -eq 0 ]]; then
  TAR_PATH="${PKG_DIR}.tar.gz"
  echo "==> 压缩 ${TAR_PATH}"
  tar -czf "${TAR_PATH}" -C "$(dirname "${PKG_DIR}")" "$(basename "${PKG_DIR}")"
  if [[ -n "${SPLIT_SIZE}" ]]; then
    echo "==> 分卷 ${SPLIT_SIZE} -> ${TAR_PATH}.part-*"
    split -b "${SPLIT_SIZE}" -d -a 3 "${TAR_PATH}" "${TAR_PATH}.part-"
    echo "    合并: cat ${TAR_PATH}.part-* > ${PKG_BASENAME}.tar.gz"
  fi
fi

echo ""
echo "完成: ${PKG_DIR}"
[[ -n "${TAR_PATH}" ]] && echo "  tar: ${TAR_PATH} ($(du -h "${TAR_PATH}" | awk '{print $1}'))"
