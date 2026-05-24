#!/usr/bin/env bash
# 下载 MediaMTX 到 scripts/.local/（不提交 git）
set -euo pipefail

VERSION="v1.11.3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="${SCRIPT_DIR}/.local"
ARCHIVE="mediamtx_${VERSION}_linux_amd64.tar.gz"
URL="https://github.com/bluenviron/mediamtx/releases/download/${VERSION}/${ARCHIVE}"

mkdir -p "${DEST}"
cd "${DEST}"

echo "下载 MediaMTX ${VERSION} ..."
curl -fsSL -o "${ARCHIVE}" "${URL}"
tar -xzf "${ARCHIVE}"
chmod +x mediamtx
rm -f "${ARCHIVE}"

echo "已安装: ${DEST}/mediamtx"
"${DEST}/mediamtx" --version
