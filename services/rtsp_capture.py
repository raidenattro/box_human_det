"""RTSP 低延迟采帧：OpenCV 或 FFmpeg（硬件解码自适应）。"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from typing import Tuple

import cv2
import numpy as np

from services.hwaccel_probe import probe_ffmpeg_decode_profile, probe_summary

_DEFAULT_FFMPEG_OPTS = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0"
_LOW_LATENCY_FFMPEG_OPTS = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", _DEFAULT_FFMPEG_OPTS)
_DRAIN_MAX = max(2, int(os.environ.get("RTSP_DRAIN_MAX", "12")))


def apply_low_latency_ffmpeg_env() -> None:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _LOW_LATENCY_FFMPEG_OPTS


def _backend_mode() -> str:
    mode = os.environ.get("RTSP_CAPTURE_BACKEND", "auto").strip().lower()
    if mode in ("opencv", "ffmpeg"):
        return mode
    profile = probe_ffmpeg_decode_profile()
    if profile.name != "software":
        return "ffmpeg"
    if os.environ.get("RTSP_FFMPEG_FORCE", "").strip().lower() in ("1", "true", "yes"):
        return "ffmpeg"
    return "opencv"


class _FfmpegRtspCapture:
    """子进程 FFmpeg 解码，后台线程只保留最新一帧。"""

    def __init__(self, url: str):
        self.url = url
        self._profile = probe_ffmpeg_decode_profile()
        self._width = 0
        self._height = 0
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._latest: np.ndarray | None = None
        self._opened = False
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()

    def isOpened(self) -> bool:
        return self._opened

    def _probe_size(self) -> tuple[int, int]:
        ffprobe = os.environ.get("FFPROBE_BIN", "ffprobe").strip() or "ffprobe"
        try:
            proc = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-rtsp_transport",
                    "tcp",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "csv=p=0:s=x",
                    self.url,
                ],
                capture_output=True,
                text=True,
                timeout=12.0,
                check=False,
            )
            line = (proc.stdout or "").strip().splitlines()[0] if proc.stdout else ""
            if "x" in line:
                w, h = line.split("x", 1)
                return max(2, int(w)), max(2, int(h))
        except Exception:
            pass
        return 640, 360

    def _build_cmd(self) -> list[str]:
        ffmpeg = os.environ.get("FFMPEG_BIN", "ffmpeg").strip() or "ffmpeg"
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-rtsp_transport",
            "tcp",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-max_delay",
            "0",
        ]
        cmd.extend(self._profile.input_args)
        if self._profile.video_codec:
            cmd.extend(["-c:v", self._profile.video_codec])
        cmd.extend(["-i", self.url, "-an", "-vf", self._profile.output_vf])
        cmd.extend(
            [
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ]
        )
        return cmd

    def open(self) -> bool:
        self._width, self._height = self._probe_size()
        cmd = self._build_cmd()
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**7,
            )
        except Exception:
            return False
        self._opened = True
        self._reader = threading.Thread(target=self._read_loop, name="ffmpeg-rtsp-reader", daemon=True)
        self._reader.start()
        return True

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        frame_bytes = self._width * self._height * 3
        stdout = self._proc.stdout
        while not self._stop.is_set():
            buf = stdout.read(frame_bytes)
            if not buf or len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape((self._height, self._width, 3))
            with self._lock:
                self._latest = frame.copy()

    def grab(self) -> bool:
        with self._lock:
            return self._latest is not None

    def retrieve(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._latest is None:
                return False, None
            return True, self._latest.copy()

    def read(self) -> tuple[bool, np.ndarray | None]:
        deadline = time.time() + float(os.environ.get("RTSP_OPEN_TIMEOUT_SEC", "8"))
        while time.time() < deadline:
            ok, frame = self.retrieve()
            if ok and frame is not None:
                return True, frame
            time.sleep(0.02)
        return False, None

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._height)
        if prop == cv2.CAP_PROP_FPS:
            return 25.0
        return 0.0

    def release(self) -> None:
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=2.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        self._opened = False


class _OpencvCaptureAdapter:
    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap

    def isOpened(self) -> bool:
        return self._cap.isOpened()

    def grab(self) -> bool:
        return self._cap.grab()

    def retrieve(self) -> tuple[bool, np.ndarray | None]:
        return self._cap.retrieve()

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self._cap.read()

    def get(self, prop: int) -> float:
        return float(self._cap.get(prop))

    def set(self, prop: int, value: float) -> bool:
        return bool(self._cap.set(prop, value))

    def release(self) -> None:
        self._cap.release()


def open_rtsp_capture(url: str, buffer_size: int = 1):
    mode = _backend_mode()
    if mode == "ffmpeg":
        cap = _FfmpegRtspCapture(url)
        if cap.open():
            print(f"ℹ️ RTSP 采帧: ffmpeg ({probe_summary()})")
            return cap
        cap.release()
        print("⚠️ FFmpeg RTSP 打开失败，回退 OpenCV")

    apply_low_latency_ffmpeg_env()
    cv_cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        cv_cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(buffer_size)))
    except Exception:
        pass
    print(f"ℹ️ RTSP 采帧: opencv-ffmpeg ({probe_summary()})")
    return _OpencvCaptureAdapter(cv_cap)


def drain_capture_buffer(cap, max_grabs: int | None = None) -> int:
    limit = _DRAIN_MAX if max_grabs is None else max(1, int(max_grabs))
    grabbed = 0
    for _ in range(limit):
        if not cap.grab():
            break
        grabbed += 1
    return grabbed


def read_latest_frame(cap, max_drain: int | None = None) -> Tuple[bool, np.ndarray | None, float]:
    limit = _DRAIN_MAX if max_drain is None else max(1, int(max_drain))
    grabbed = False
    for _ in range(limit):
        if not cap.grab():
            break
        grabbed = True
    if not grabbed:
        ok, frame = cap.read()
        return ok, frame, time.time()
    ok, frame = cap.retrieve()
    return ok, frame, time.time()


def read_rtsp_frame_once(url: str, timeout_sec: float | None = None) -> np.ndarray | None:
    """低频抓一帧（UI 缩略图等）。Docker 内 FFmpeg 硬解常能连上 RTSP 但读不到帧，此处固定 OpenCV。"""
    if timeout_sec is not None:
        os.environ["RTSP_OPEN_TIMEOUT_SEC"] = str(timeout_sec)
    prev_backend = os.environ.get("RTSP_CAPTURE_BACKEND")
    os.environ["RTSP_CAPTURE_BACKEND"] = "opencv"
    try:
        cap = open_rtsp_capture(url, buffer_size=1)
        try:
            if not cap.isOpened():
                return None
            ok, frame, _ = read_latest_frame(cap, max_drain=4)
            return frame if ok and frame is not None else None
        finally:
            cap.release()
    finally:
        if prev_backend is None:
            os.environ.pop("RTSP_CAPTURE_BACKEND", None)
        else:
            os.environ["RTSP_CAPTURE_BACKEND"] = prev_backend
