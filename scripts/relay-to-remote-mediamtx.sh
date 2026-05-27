#!/usr/bin/env bash
# 本机 MediaMTX/MP4 → 远程 MediaMTX（153 等）推流
# 用法:
#   REMOTE_HOST=192.168.1.153 ./scripts/relay-to-remote-mediamtx.sh start
#   ./scripts/relay-to-remote-mediamtx.sh stop
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_DIR="${SCRIPT_DIR}/.local/relay-remote"
REMOTE_HOST="${REMOTE_HOST:-192.168.1.153}"
REMOTE_PORT="${REMOTE_PORT:-8554}"
LOCAL_RTSP="${LOCAL_RTSP:-rtsp://127.0.0.1:8554}"
VIDEO="${VIDEO:-${ROOT}/test/c7e2fb9b6551c2fc6a4c93ee65ac6c23_raw.mp4}"

# 本机已有 cam1-8 → 中继到远程同 path
RELAY_PATHS=(cam1 cam2 cam3 cam4 cam5 cam6 cam7 cam8)
# 远程无本机源 → MP4 直推
PUBLISH_PATHS=(cam9 cam10 cam11 12 ces)

_remote_rtsp() {
  echo "rtsp://${REMOTE_HOST}:${REMOTE_PORT}/$1"
}

_start_relay() {
  local path="$1"
  local pidf="${LOCAL_DIR}/relay-${path}.pid"
  local logf="${LOCAL_DIR}/relay-${path}.log"
  if [[ -f "${pidf}" ]] && kill -0 "$(cat "${pidf}")" 2>/dev/null; then
    echo "已在运行: ${path} (pid=$(cat "${pidf}"))"
    return 0
  fi
  nohup ffmpeg -hide_banner -loglevel warning -rtsp_transport tcp \
    -i "${LOCAL_RTSP}/${path}" \
    -an -c:v libx264 -pix_fmt yuv420p -preset ultrafast -tune zerolatency \
    -g 25 -max_delay 0 -f rtsp -rtsp_transport tcp "$(_remote_rtsp "${path}")" \
    >"${logf}" 2>&1 &
  echo $! >"${pidf}"
  sleep 1
  if kill -0 "$(cat "${pidf}")" 2>/dev/null; then
    echo "中继已启动: ${LOCAL_RTSP}/${path} → $(_remote_rtsp "${path}")"
  else
    echo "中继失败: ${path}，见 ${logf}"
    tail -5 "${logf}" 2>/dev/null || true
    return 1
  fi
}

_start_publish() {
  local path="$1"
  local pidf="${LOCAL_DIR}/pub-${path}.pid"
  local logf="${LOCAL_DIR}/pub-${path}.log"
  if [[ ! -f "${VIDEO}" ]]; then
    echo "视频不存在: ${VIDEO}"
    return 1
  fi
  if [[ -f "${pidf}" ]] && kill -0 "$(cat "${pidf}")" 2>/dev/null; then
    echo "已在运行: ${path} (pid=$(cat "${pidf}"))"
    return 0
  fi
  nohup ffmpeg -hide_banner -loglevel warning -re -stream_loop -1 \
    -i "${VIDEO}" \
    -vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2" \
    -r 25 -c:v libx264 -pix_fmt yuv420p -preset ultrafast -tune zerolatency \
    -b:v 2500k -maxrate 2500k -bufsize 5000k -g 25 -an \
    -f rtsp -rtsp_transport tcp "$(_remote_rtsp "${path}")" \
    >"${logf}" 2>&1 &
  echo $! >"${pidf}"
  sleep 1
  if kill -0 "$(cat "${pidf}")" 2>/dev/null; then
    echo "直推已启动: ${path} → $(_remote_rtsp "${path}")"
  else
    echo "直推失败: ${path}，见 ${logf}"
    tail -5 "${logf}" 2>/dev/null || true
    return 1
  fi
}

_stop_pattern() {
  local prefix="$1"
  shopt -s nullglob
  for pidf in "${LOCAL_DIR}/${prefix}"-*.pid; do
    [[ -f "${pidf}" ]] || continue
    local pid
    pid="$(cat "${pidf}")"
    kill "${pid}" 2>/dev/null || true
    rm -f "${pidf}"
  done
}

cmd="${1:-start}"
mkdir -p "${LOCAL_DIR}"

case "${cmd}" in
  start)
    echo "远程 MediaMTX: ${REMOTE_HOST}:${REMOTE_PORT}"
    for p in "${RELAY_PATHS[@]}"; do _start_relay "${p}" || true; done
    for p in "${PUBLISH_PATHS[@]}"; do _start_publish "${p}" || true; done
    echo "完成。cam13 由 153 从 192.168.1.149 拉流，无需本机推。"
    ;;
  stop)
    _stop_pattern relay
    _stop_pattern pub
    echo "已停止远程推流/中继进程"
    ;;
  status)
    for pidf in "${LOCAL_DIR}"/*.pid; do
      [[ -f "${pidf}" ]] || continue
      name="$(basename "${pidf}" .pid)"
      pid="$(cat "${pidf}")"
      if kill -0 "${pid}" 2>/dev/null; then
        echo "running ${name} pid=${pid}"
      else
        echo "dead ${name}"
      fi
    done
    ;;
  *)
    echo "用法: REMOTE_HOST=192.168.1.153 $0 {start|stop|status}"
    exit 1
    ;;
esac
