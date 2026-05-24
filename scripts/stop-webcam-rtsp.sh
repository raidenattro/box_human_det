#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="${SCRIPT_DIR}/.local/mediamtx.pid"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "MediaMTX 未在运行"
  exit 0
fi

PID="$(cat "${PID_FILE}")"
if kill -0 "${PID}" 2>/dev/null; then
  kill "${PID}" 2>/dev/null || true
  sleep 1
  kill -9 "${PID}" 2>/dev/null || true
fi
rm -f "${PID_FILE}"
pkill -f "ffmpeg.*rtsp://127.0.0.1:8554/cam" 2>/dev/null || true
echo "已停止 MediaMTX / ffmpeg 推流"
