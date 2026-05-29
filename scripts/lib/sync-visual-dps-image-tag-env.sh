#!/usr/bin/env bash
# 将 VISUAL_DPS_IMAGE_TAG 写入项目 .env，供 compose up / 页脚版本展示使用。
set -euo pipefail

sync_visual_dps_image_tag_env() {
  local root="$1"
  local tag="$2"
  local env_file="${root}/.env"

  if [[ -z "${tag}" ]]; then
    echo "sync_visual_dps_image_tag_env: 空 tag" >&2
    return 1
  fi

  if [[ -f "${env_file}" ]]; then
    if grep -q '^VISUAL_DPS_IMAGE_TAG=' "${env_file}"; then
      sed -i "s|^VISUAL_DPS_IMAGE_TAG=.*|VISUAL_DPS_IMAGE_TAG=${tag}|" "${env_file}"
    else
      printf '\nVISUAL_DPS_IMAGE_TAG=%s\n' "${tag}" >> "${env_file}"
    fi
  else
    printf 'VISUAL_DPS_IMAGE_TAG=%s\n' "${tag}" > "${env_file}"
  fi
  echo "  已写入 ${env_file}: VISUAL_DPS_IMAGE_TAG=${tag}"
}
