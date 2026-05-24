#!/usr/bin/env bash
# 将本地 MP4 循环推送到 MediaMTX（每路 path 独立进程）
# 用法:
#   ./scripts/start-mp4-rtsp.sh
#   ./scripts/start-mp4-rtsp.sh /path/to/video.mp4 cam2
#   ./scripts/start-mp4-rtsp-multi.sh /path/to/video.mp4 cam2 cam3 cam4 cam5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MTX_PID="${SCRIPT_DIR}/.local/mediamtx.pid"

DEFAULT_VIDEO="/mnt/c/Users/sugar/Videos/33611ddf17439fe92fa3620b1fe6da92.mp4"
VIDEO="${1:-${DEFAULT_VIDEO}}"
PATH_NAME="${2:-cam2}"
RTSP_URL="rtsp://127.0.0.1:8554/${PATH_NAME}"

PID_FILE="${SCRIPT_DIR}/.local/ffmpeg-mp4-rtsp-${PATH_NAME}.pid"
LOG_FILE="${SCRIPT_DIR}/.local/ffmpeg-mp4-rtsp-${PATH_NAME}.log"

if [[ ! -f "${VIDEO}" ]]; then
  echo "视频文件不存在: ${VIDEO}"
  exit 1
fi

if ! command -v ffmpeg >/dev/null; then
  echo "未找到 ffmpeg，请先安装: sudo apt install ffmpeg"
  exit 1
fi

if [[ ! -f "${MTX_PID}" ]] || ! kill -0 "$(cat "${MTX_PID}")" 2>/dev/null; then
  echo "MediaMTX 未运行，正在启动…"
  "${SCRIPT_DIR}/start-webcam-rtsp.sh"
fi

if command -v curl >/dev/null; then
  curl -fsS -X POST "http://127.0.0.1:9997/v3/config/paths/replace/${PATH_NAME}" \
    -H 'Content-Type: application/json' \
    -d '{"source":"publisher"}' >/dev/null 2>&1 \
    || curl -fsS -X POST "http://127.0.0.1:9997/v3/config/paths/add/${PATH_NAME}" \
      -H 'Content-Type: application/json' \
      -d '{"source":"publisher"}' >/dev/null 2>&1 \
    || true
fi

if [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "MP4 推流已在运行: ${PATH_NAME} (pid=$(cat "${PID_FILE}"))"
  echo "RTSP: ${RTSP_URL}"
  exit 0
fi

mkdir -p "${SCRIPT_DIR}/.local"

nohup ffmpeg -hide_banner -loglevel warning -re -stream_loop -1 \
  -i "${VIDEO}" \
  -vf "scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2" \
  -r 15 \
  -c:v libx264 -pix_fmt yuv420p -preset ultrafast -tune zerolatency \
  -b:v 800k -maxrate 800k -bufsize 1600k \
  -g 15 -f rtsp -rtsp_transport tcp \
  "${RTSP_URL}" \
  >"${LOG_FILE}" 2>&1 &
echo $! >"${PID_FILE}"
sleep 2

if ! kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
  echo "推流启动失败 (${PATH_NAME})，日志:"
  tail -20 "${LOG_FILE}" || true
  exit 1
fi

echo "MP4 推流已启动: ${PATH_NAME} (pid=$(cat "${PID_FILE}"))"
echo "源文件: ${VIDEO}"
echo "RTSP: ${RTSP_URL}"
echo "日志: ${LOG_FILE}"
