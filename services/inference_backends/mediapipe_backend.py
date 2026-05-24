"""MediaPipe Pose Landmarker 轻量后端（本地测试平替，CPU 友好）。"""

from __future__ import annotations

import asyncio
import os
import urllib.request

import cv2
import numpy as np

from services.inference_backends.base import PoseBatch

# MediaPipe 33 点 → COCO 17 点（与 inference_service 手腕/肩索引一致）
_MP_TO_COCO = (
    (0, 0),
    (2, 1),
    (5, 2),
    (7, 3),
    (8, 4),
    (11, 5),
    (12, 6),
    (13, 7),
    (14, 8),
    (15, 9),
    (16, 10),
    (23, 11),
    (24, 12),
    (25, 13),
    (26, 14),
    (27, 15),
    (28, 16),
)

_DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)


def _resolve_model_path(app_config: dict) -> str:
    models_cfg = app_config.get("models", {}) if isinstance(app_config.get("models"), dict) else {}
    raw = str(models_cfg.get("mediapipe_model_path") or "").strip()
    if raw:
        return raw
    base = app_config.get("paths", {}).get("base_localdata_dir", "localdata")
    return os.path.join(base, "models", "pose_landmarker_lite.task")


def _ensure_model_file(model_path: str, url: str) -> None:
    if os.path.isfile(model_path):
        return
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    print(f"⬇️ 正在下载 MediaPipe 模型: {url}")
    urllib.request.urlretrieve(url, model_path)


class MediaPipeBackend:
    name = "mediapipe"

    def __init__(self, app_config: dict, executor):
        self.app_config = app_config
        self._executor = executor
        self._landmarker = None
        self._last_bboxes: np.ndarray | None = None
        self._last_pose: PoseBatch = PoseBatch.empty()

    def ensure_loaded(self) -> None:
        if self._landmarker is not None:
            return

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        models_cfg = self.app_config.get("models", {})
        model_path = _resolve_model_path(self.app_config)
        model_url = str(models_cfg.get("mediapipe_model_url") or _DEFAULT_MODEL_URL).strip()
        _ensure_model_file(model_path, model_url)

        num_poses = int(models_cfg.get("mediapipe_num_poses", 4) or 4)
        num_poses = max(1, min(num_poses, 8))

        print(f"🚀 正在加载 MediaPipe Pose（lite，最多 {num_poses} 人）...")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=0.4,
            min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._mp_image_cls = mp.Image
        self._mp_image_format = mp.ImageFormat.SRGB

    def _process_sync(self, frame) -> tuple[np.ndarray, PoseBatch]:
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp_image_cls(image_format=self._mp_image_format, data=rgb)
        result = self._landmarker.detect(mp_image)

        bboxes = []
        keypoints_list = []
        scores_list = []

        for landmarks in result.pose_landmarks or []:
            kpts = np.zeros((17, 2), dtype=np.float32)
            scores = np.zeros(17, dtype=np.float32)
            for mp_idx, coco_idx in _MP_TO_COCO:
                if mp_idx >= len(landmarks):
                    continue
                lm = landmarks[mp_idx]
                kpts[coco_idx, 0] = float(lm.x) * w
                kpts[coco_idx, 1] = float(lm.y) * h
                vis = getattr(lm, "visibility", None)
                if vis is None:
                    vis = getattr(lm, "presence", 1.0)
                scores[coco_idx] = float(vis or 0.0)

            valid = scores > 0.15
            if not np.any(valid):
                continue

            xs = kpts[valid, 0]
            ys = kpts[valid, 1]
            pad = 8.0
            x1 = max(0.0, float(xs.min()) - pad)
            y1 = max(0.0, float(ys.min()) - pad)
            x2 = min(float(w), float(xs.max()) + pad)
            y2 = min(float(h), float(ys.max()) + pad)
            bboxes.append([x1, y1, x2, y2])
            keypoints_list.append(kpts)
            scores_list.append(scores)

        if not bboxes:
            empty = PoseBatch.empty()
            return np.empty((0, 4), dtype=np.float32), empty

        return (
            np.asarray(bboxes, dtype=np.float32),
            PoseBatch(
                keypoints=np.stack(keypoints_list, axis=0),
                keypoint_scores=np.stack(scores_list, axis=0),
            ),
        )

    async def detect_bboxes(self, frame) -> np.ndarray:
        self.ensure_loaded()
        loop = asyncio.get_running_loop()
        bboxes, pose_batch = await loop.run_in_executor(self._executor, self._process_sync, frame)
        self._last_bboxes = bboxes
        self._last_pose = pose_batch
        return bboxes

    async def estimate_pose(self, frame, bboxes: np.ndarray) -> PoseBatch:
        if bboxes is None or len(bboxes) == 0:
            return PoseBatch.empty()
        if (
            self._last_bboxes is not None
            and len(self._last_bboxes) == len(bboxes)
            and self._last_pose.num_persons > 0
        ):
            return self._last_pose

        self.ensure_loaded()
        loop = asyncio.get_running_loop()
        _, pose_batch = await loop.run_in_executor(self._executor, self._process_sync, frame)
        self._last_pose = pose_batch
        return pose_batch
