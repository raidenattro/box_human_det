"""事件 Worker 水平扩展：按 camera_id 分片，避免 Pub/Sub 多实例重复处理。"""

from __future__ import annotations

import os
import zlib


def shard_config() -> tuple[int, int]:
    """返回 (shard_count, shard_index)，默认单实例处理全部摄像头。"""
    try:
        count = int(os.environ.get("EVENT_WORKER_SHARD_COUNT", "1") or "1")
    except ValueError:
        count = 1
    try:
        index = int(os.environ.get("EVENT_WORKER_SHARD_INDEX", "0") or "0")
    except ValueError:
        index = 0
    count = max(1, count)
    index = max(0, min(index, count - 1))
    return count, index


def camera_shard_id(camera_id: str) -> int:
    cid = str(camera_id or "").strip()
    if not cid:
        return 0
    return zlib.crc32(cid.encode("utf-8")) & 0xFFFFFFFF


def owns_camera(camera_id: str, shard_count: int | None = None, shard_index: int | None = None) -> bool:
    """该实例是否负责此 camera_id（同一摄像头始终落在同一分片）。"""
    count, index = shard_config()
    if shard_count is not None:
        count = max(1, int(shard_count))
    if shard_index is not None:
        index = max(0, min(int(shard_index), count - 1))
    if count <= 1:
        return True
    cid = str(camera_id or "").strip()
    if not cid:
        return False
    return (camera_shard_id(cid) % count) == index


def shard_label() -> str:
    count, index = shard_config()
    if count <= 1:
        return "shard=all"
    return f"shard={index}/{count}"
