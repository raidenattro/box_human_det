"""可插拔推理后端：mediapipe | rtmpose_onnx（后续 yolo_pose）。"""

from __future__ import annotations

import os

BACKEND_MEDIAPIPE = "mediapipe"
BACKEND_RTMPOSE_ONNX = "rtmpose_onnx"
_LITE_BACKENDS = frozenset({BACKEND_MEDIAPIPE, BACKEND_RTMPOSE_ONNX})
_ALIASES = {
    "lite": BACKEND_MEDIAPIPE,
    "mp": BACKEND_MEDIAPIPE,
    "mediapipe": BACKEND_MEDIAPIPE,
    "rtmpose_onnx": BACKEND_RTMPOSE_ONNX,
    "rtmpose-t": BACKEND_RTMPOSE_ONNX,
    "rtmpose_t": BACKEND_RTMPOSE_ONNX,
    "rtmpose-cpu": BACKEND_RTMPOSE_ONNX,
    "rtmpose_cpu": BACKEND_RTMPOSE_ONNX,
    "default": BACKEND_RTMPOSE_ONNX,
}


def resolve_backend_name(
    app_config: dict | None = None,
    overrides: dict | None = None,
) -> str:
    """优先级：摄像头/全局运行时配置 overrides > 环境变量 > app_config.json。"""

    def _from_raw(raw: str) -> str | None:
        key = (raw or "").strip().lower()
        if not key:
            return None
        name = _ALIASES.get(key, key)
        if name in (BACKEND_MEDIAPIPE, BACKEND_RTMPOSE_ONNX):
            return name
        return None

    if isinstance(overrides, dict):
        name = _from_raw(str(overrides.get("models.backend", "")))
        if name:
            return name

    env_raw = os.environ.get("INFERENCE_BACKEND", "").strip().lower()
    name = _from_raw(env_raw)
    if name:
        return name

    if isinstance(app_config, dict):
        models = app_config.get("models")
        if isinstance(models, dict):
            name = _from_raw(str(models.get("backend", "")))
            if name:
                return name
    return BACKEND_RTMPOSE_ONNX


def create_inference_backend(app_config: dict, executor):
    name = resolve_backend_name(app_config)
    if name == BACKEND_MEDIAPIPE:
        from services.inference_backends.mediapipe_backend import MediaPipeBackend

        return MediaPipeBackend(app_config, executor)
    if name == BACKEND_RTMPOSE_ONNX:
        from services.inference_backends.rtmpose_onnx_backend import RTMPoseOnnxBackend

        return RTMPoseOnnxBackend(app_config, executor)
    raise RuntimeError(f"未知推理后端: {name}，可选 mediapipe | rtmpose_onnx")
