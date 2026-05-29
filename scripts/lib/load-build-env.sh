#!/usr/bin/env bash
# 构建前加载 .env 中的国内镜像/代理；剔除 WSL 专用无效代理
# shellcheck disable=SC2034
load_build_env() {
  local root="${1:-.}"
  if [[ -f "${root}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${root}/.env"
    set +a
  fi

  export APT_MIRROR="${APT_MIRROR:-mirrors.aliyun.com}"
  export PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
  export GITHUB_PROXY_BASE="${GITHUB_PROXY_BASE:-}"
  export OPENMMLAB_MIRROR_BASE="${OPENMMLAB_MIRROR_BASE:-}"

  # 构建专用代理（153 等 Linux 机可设；优先于下面自动清理后的 HTTP_PROXY）
  if [[ -n "${BUILD_HTTP_PROXY:-}" ]]; then
    export HTTP_PROXY="${BUILD_HTTP_PROXY}"
    export HTTPS_PROXY="${BUILD_HTTPS_PROXY:-${BUILD_HTTP_PROXY}}"
    export http_proxy="${HTTP_PROXY}"
    export https_proxy="${HTTPS_PROXY}"
  else
    # 从 .env 继承的 WSL 代理在 Linux 宿主机上无效，避免 docker build 卡死
    if [[ "${HTTP_PROXY:-}" == *"172.26."* ]] || [[ "${HTTPS_PROXY:-}" == *"172.26."* ]]; then
      unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
    fi
  fi

  export DOCKER_BUILDKIT=1
}
