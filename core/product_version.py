"""产品版本号（与 web/version.json 同步，由 scripts/bump-build.mjs 维护）。"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_VERSION_CANDIDATES = (
    Path(os.environ.get("VISUAL_DPS_VERSION_FILE", "")),
    _REPO_ROOT / "version.json",
    _REPO_ROOT / "web" / "version.json",
)


@lru_cache(maxsize=1)
def load_version_dict() -> dict:
    for path in _VERSION_CANDIDATES:
        if not path or not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            continue
    return {"major": 0, "minor": 0, "patch": 0, "build": 0}


def format_product_version(data: dict | None = None) -> str:
    v = data if isinstance(data, dict) else load_version_dict()
    major = int(v.get("major", 0))
    minor = int(v.get("minor", 0))
    patch = int(v.get("patch", 0))
    build = int(v.get("build", 0))
    return f"v{major}.{minor}.{patch}.build{build}"


def short_image_ref(image: str) -> str:
    """镜像全名 → 仅展示 tag（如 20260528-120000-abc1234）。"""
    s = str(image or "").strip()
    if not s or s == "—":
        return "—"
    if "@" in s:
        s = s.split("@", 1)[0]
    if ":" in s:
        return s.rsplit(":", 1)[-1][:48]
    return s[:48]
