#!/usr/bin/env bash
# 在 WSL 内启动 MediaMTX，并由其拉起 ffmpeg 从 /dev/video0 推流到 rtsp://127.0.0.1:8554/cam
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_BIN="${SCRIPT_DIR}/.local/mediamtx"
GENERATED_CONFIG="${ROOT}/localdata/mediamtx.yml"
CONFIG="${SCRIPT_DIR}/mediamtx.yml"
# 仅当自动生成配置已启用 HLS/WebRTC 时才使用（避免旧版 root 写入的空地址配置）
if [[ -f "${GENERATED_CONFIG}" ]] && grep -qE 'hlsAddress: :|webrtcAddress: :' "${GENERATED_CONFIG}" 2>/dev/null; then
  CONFIG="${GENERATED_CONFIG}"
elif [[ -f "${GENERATED_CONFIG}" ]]; then
  echo "提示: ${GENERATED_CONFIG} 未启用 HLS/WebRTC，使用 ${CONFIG}"
fi
PID_FILE="${SCRIPT_DIR}/.local/mediamtx.pid"
LOG_FILE="${SCRIPT_DIR}/.local/mediamtx.log"

if [[ ! -x "${LOCAL_BIN}" ]]; then
  echo "未找到 MediaMTX，请先运行: ${SCRIPT_DIR}/install-mediamtx.sh"
  exit 1
fi

if [[ ! -e /dev/video0 ]]; then
  echo "未找到 /dev/video0。请先在 Windows 执行 usbipd attach 挂载摄像头，例如:"
  echo '  usbipd bind --busid 2-8 && usbipd attach --wsl --busid 2-8'
  exit 1
fi

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "MediaMTX 已在运行 (pid=$(cat "${PID_FILE}"))"
  echo "RTSP: rtsp://127.0.0.1:8554/cam"
  exit 0
fi

mkdir -p "${SCRIPT_DIR}/.local"
cd "${SCRIPT_DIR}/.local"

nohup "${LOCAL_BIN}" "${CONFIG}" >"${LOG_FILE}" 2>&1 &
echo $! >"${PID_FILE}"
sleep 2

if ! kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "启动失败，日志:"
  tail -30 "${LOG_FILE}" || true
  exit 1
fi

echo "MediaMTX 已启动 (pid=$(cat "${PID_FILE}"))"
echo "RTSP 地址: rtsp://127.0.0.1:8554/cam"
echo "日志: ${LOG_FILE}"
echo ""
echo "visual-dps app_config.json 可配置:"
echo '  "stream_url": "rtsp://127.0.0.1:8554/cam"'
