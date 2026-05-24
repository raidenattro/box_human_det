#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/.local/ffmpeg-mp4-rtsp.pid"
if [[ -f "${PID_FILE}" ]]; then
  pid="$(cat "${PID_FILE}")"
  kill "${pid}" 2>/dev/null || true
  rm -f "${PID_FILE}"
  echo "已停止 MP4 推流 (pid=${pid})"
else
  pkill -f "ffmpeg.*8554/cam2" 2>/dev/null || true
  echo "已尝试停止 cam2 推流"
fi
