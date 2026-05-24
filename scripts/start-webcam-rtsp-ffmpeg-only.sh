#!/usr/bin/env bash
# 不依赖 MediaMTX runOnInit：手动启动 MediaMTX + 独立 ffmpeg 推流（便于调试）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_BIN="${SCRIPT_DIR}/.local/mediamtx"
MTX_PID="${SCRIPT_DIR}/.local/mediamtx.pid"
FFMPEG_PID="${SCRIPT_DIR}/.local/ffmpeg-rtsp.pid"

"${SCRIPT_DIR}/start-webcam-rtsp.sh" 2>/dev/null || true

# 若 runOnInit 方式失败，可改用此脚本单独推流
if [[ -f "${MTX_PID}" ]] && pgrep -f "ffmpeg.*8554/cam" >/dev/null; then
  echo "推流已在运行"
  exit 0
fi

if [[ ! -e /dev/video0 ]]; then
  echo "未找到 /dev/video0"
  exit 1
fi

# 确保 MediaMTX 在跑（仅作 RTSP 服务器）
if [[ ! -f "${MTX_PID}" ]] || ! kill -0 "$(cat "${MTX_PID}")" 2>/dev/null; then
  mkdir -p "${SCRIPT_DIR}/.local"
  cat > "${SCRIPT_DIR}/.local/mediamtx-min.yml" <<'EOF'
logLevel: info
rtspAddress: :8554
paths:
  cam:
    source: publisher
EOF
  cd "${SCRIPT_DIR}/.local"
  nohup "${LOCAL_BIN}" mediamtx-min.yml >mediamtx.log 2>&1 &
  echo $! >"${MTX_PID}"
  sleep 1
fi

nohup ffmpeg -hide_banner -loglevel warning \
  -f v4l2 -input_format mjpeg -video_size 640x480 -framerate 15 \
  -i /dev/video0 \
  -c:v libx264 -pix_fmt yuv420p -preset ultrafast -tune zerolatency \
  -b:v 800k -g 15 -f rtsp -rtsp_transport tcp \
  rtsp://127.0.0.1:8554/cam \
  >"${SCRIPT_DIR}/.local/ffmpeg-rtsp.log" 2>&1 &
echo $! >"${FFMPEG_PID}"
sleep 2
echo "ffmpeg 推流 pid=$(cat "${FFMPEG_PID}")"
echo "RTSP: rtsp://127.0.0.1:8554/cam"
