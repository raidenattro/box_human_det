#!/usr/bin/env bash
# 离线包 dry 校验（不 install、不 load 镜像）
# 用法: ./deploy/verify-package.sh [包根目录]
set -euo pipefail

PKG_ROOT="$(cd "${1:-.}" && pwd)"
FAIL=0

note_fail() { echo "FAIL: $*" >&2; FAIL=1; }
note_ok() { echo "OK: $*"; }

echo "==> 校验离线包: ${PKG_ROOT}"

for f in install.sh docker-images/bundle.tar app/docker-compose.deploy.yml app/.env app/app_config.json; do
  if [[ -f "${PKG_ROOT}/${f}" ]]; then
    note_ok "${f}"
  else
    note_fail "缺少 ${f}"
  fi
done

if [[ -f "${PKG_ROOT}/OFFLINE-QUICKSTART.md" ]]; then
  note_ok "OFFLINE-QUICKSTART.md"
else
  note_fail "缺少 OFFLINE-QUICKSTART.md"
fi

if [[ -f "${PKG_ROOT}/docker-images/bundle.tar" ]]; then
  if tar -tf "${PKG_ROOT}/docker-images/bundle.tar" repositories >/dev/null 2>&1; then
    note_ok "bundle.tar (docker save)"
    echo "--- 镜像列表 ---"
    tar -xOf "${PKG_ROOT}/docker-images/bundle.tar" repositories | python3 -m json.tool 2>/dev/null || true
  else
    note_fail "bundle.tar 无法读取 repositories"
  fi
fi

# 分卷包
shopt -s nullglob
parts=("${PKG_ROOT}"/docker-images/bundle.tar.part-*)
if [[ ${#parts[@]} -gt 0 ]]; then
  note_ok "发现分卷 bundle.tar.part-* (${#parts[@]} 片)"
fi

if [[ -d "${PKG_ROOT}/app/localdata/models" ]]; then
  # shellcheck disable=SC1091
  source "${PKG_ROOT}/app/deploy/check-model-weights.sh"
  if visual_dps_check_model_weights "${PKG_ROOT}/app/localdata"; then
    note_ok "模型权重完整"
  else
    note_fail "模型权重不完整"
  fi
else
  note_fail "缺少 app/localdata/models"
fi

if [[ -f "${PKG_ROOT}/app/deploy/model-weights-spec.sh" ]]; then
  note_ok "model-weights-spec.sh"
else
  note_fail "缺少 model-weights-spec.sh"
fi

if [[ "${FAIL}" -ne 0 ]]; then
  echo "校验未通过" >&2
  exit 1
fi
echo "==> 全部校验通过"
