"""线程安全「仅保留最新帧」缓冲，供 RTSP 后台读帧与推理快照消费。"""

from __future__ import annotations

import threading
import time

import numpy as np


class LatestFrameBuffer:
    """后台读帧线程写入；推理侧通过 snapshot() 取副本，互不阻塞解码。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._captured_at: float = 0.0
        self._seq: int = 0

    def update(self, frame: np.ndarray) -> None:
        with self._lock:
            self._latest = frame
            self._captured_at = time.time()
            self._seq += 1

    def snapshot(self) -> tuple[bool, np.ndarray | None, float, int]:
        with self._lock:
            if self._latest is None:
                return False, None, 0.0, self._seq
            return True, self._latest.copy(), self._captured_at, self._seq

    @property
    def seq(self) -> int:
        with self._lock:
            return self._seq
