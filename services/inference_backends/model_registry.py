"""推理模型预设：models.backend 存 preset id（如 rtmpose_t、yolo26s_pose）。"""

from __future__ import annotations

from dataclasses import dataclass

BACKEND_RTMPOSE_ONNX = "rtmpose_onnx"
BACKEND_YOLO_POSE = "yolo_pose"

_ONNX_SDK = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk"
_DET_NANO = f"{_ONNX_SDK}/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.zip"


@dataclass(frozen=True)
class ModelPreset:
    id: str
    family: str
    variant: str
    label: str
    short_label: str


RTMPoseOnnxPreset = ModelPreset  # alias


_PRESETS: dict[str, ModelPreset] = {}
_ALIASES: dict[str, str] = {
    "lite": "rtmpose_t",
    "mp": "rtmpose_t",
    "mediapipe": "rtmpose_t",
    "mmpose": "rtmpose_t",
    "mm": "rtmpose_t",
    "default": "rtmpose_t",
    "rtmpose_onnx": "rtmpose_t",
    "rtmpose-t": "rtmpose_t",
    "rtmpose_t": "rtmpose_t",
    "yolo_pose": "yolo26s_pose",
}


def _reg(p: ModelPreset) -> None:
    _PRESETS[p.id] = p


_reg(
    ModelPreset(
        "rtmpose_t",
        BACKEND_RTMPOSE_ONNX,
        "t",
        "RTMPose-T + RTMDet-nano（ONNX）",
        "RTMPose-T",
    )
)
_reg(
    ModelPreset(
        "rtmpose_s",
        BACKEND_RTMPOSE_ONNX,
        "s",
        "RTMPose-S + RTMDet-nano（ONNX）",
        "RTMPose-S",
    )
)
_reg(
    ModelPreset(
        "rtmpose_m",
        BACKEND_RTMPOSE_ONNX,
        "m",
        "RTMPose-M + RTMDet-nano（ONNX）",
        "RTMPose-M",
    )
)
for _vid, _label, _short in (
    ("n", "YOLO26n-pose", "YOLO26n"),
    ("s", "YOLO26s-pose", "YOLO26s"),
    ("m", "YOLO26m-pose", "YOLO26m"),
    ("l", "YOLO26l-pose", "YOLO26l"),
):
    _reg(
        ModelPreset(
            f"yolo26{_vid}_pose",
            BACKEND_YOLO_POSE,
            _vid,
            _label,
            _short,
        )
    )

DEFAULT_PRESET_ID = "rtmpose_t"
ALLOWED_PRESET_IDS = frozenset(_PRESETS.keys())
LITE_BACKEND_FAMILIES = frozenset({BACKEND_RTMPOSE_ONNX, BACKEND_YOLO_POSE})


def resolve_model_preset(
    app_config: dict | None = None,
    overrides: dict | None = None,
) -> ModelPreset:
    def _raw() -> str:
        if isinstance(overrides, dict):
            v = str(overrides.get("models.backend", "")).strip().lower()
            if v:
                return v
        import os

        env = os.environ.get("INFERENCE_BACKEND", "").strip().lower()
        if env:
            return env
        if isinstance(app_config, dict):
            models = app_config.get("models")
            if isinstance(models, dict):
                v = str(models.get("backend", "")).strip().lower()
                if v:
                    return v
        return ""

    key = _ALIASES.get(_raw(), _raw())
    if not key:
        return _PRESETS[DEFAULT_PRESET_ID]
    return _PRESETS.get(key, _PRESETS[DEFAULT_PRESET_ID])


def resolve_backend_family(
    app_config: dict | None = None,
    overrides: dict | None = None,
) -> str:
    return resolve_model_preset(app_config, overrides).family


def normalize_backend_setting(raw: str) -> str:
    key = _ALIASES.get(str(raw or "").strip().lower(), str(raw or "").strip().lower())
    if key not in _PRESETS:
        raise ValueError(
            f"backend must be one of: {', '.join(sorted(_PRESETS.keys()))}"
        )
    return key


# RTMPose ONNX 资产（variant -> det/pose 目录名与 zip）
RTMPOSE_VARIANT_ASSETS: dict[str, dict[str, str | tuple[int, int]]] = {
    "t": {
        "det_dir": "rtmdet_nano",
        "pose_dir": "rtmpose_t",
        "det_url": _DET_NANO,
        "pose_url": (
            f"{_ONNX_SDK}/rtmpose-t_simcc-body7_pt-body7_420e-256x192-026a1439_20230504.zip"
        ),
        "det_size": (320, 320),
        "pose_size": (192, 256),
    },
    "s": {
        "det_dir": "rtmdet_nano",
        "pose_dir": "rtmpose_s",
        "det_url": _DET_NANO,
        "pose_url": (
            f"{_ONNX_SDK}/rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip"
        ),
        "det_size": (320, 320),
        "pose_size": (192, 256),
    },
    "m": {
        "det_dir": "rtmdet_nano",
        "pose_dir": "rtmpose_m",
        "det_url": _DET_NANO,
        "pose_url": (
            f"{_ONNX_SDK}/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"
        ),
        "det_size": (320, 320),
        "pose_size": (192, 256),
    },
}

YOLO_VARIANT_WEIGHTS: dict[str, str] = {
    "n": "yolo26n-pose.pt",
    "s": "yolo26s-pose.pt",
    "m": "yolo26m-pose.pt",
    "l": "yolo26l-pose.pt",
}
