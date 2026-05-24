"""OpenMMLab MMDet + MMPose 推理后端（默认，生产路径）。"""

from __future__ import annotations

import asyncio

import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model as init_pose_model, inference_topdown
from mmpose.structures import merge_data_samples
from mmpose.utils import register_all_modules, adapt_mmdet_pipeline

from services.inference_backends.base import PoseBatch

register_all_modules()


class MMPoseBackend:
    name = "mmpose"

    def __init__(self, app_config: dict, executor):
        self.app_config = app_config
        self._executor = executor
        self.det_model = None
        self.pose_model = None

    def ensure_loaded(self) -> None:
        if self.det_model is not None and self.pose_model is not None:
            return

        print("🚀 正在加载 MMDet/MMPose 模型...")
        models_cfg = self.app_config["models"]
        device = models_cfg["device"]
        self.det_model = init_detector(
            models_cfg["det_config"],
            models_cfg["det_checkpoint"],
            device=device,
        )
        self.det_model.cfg = adapt_mmdet_pipeline(self.det_model.cfg)
        self.pose_model = init_pose_model(
            models_cfg["pose_config"],
            models_cfg["pose_checkpoint"],
            device=device,
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))),
        )

    async def detect_bboxes(self, frame) -> np.ndarray:
        loop = asyncio.get_running_loop()
        det_result = await loop.run_in_executor(
            self._executor, inference_detector, self.det_model, frame
        )
        valid = (det_result.pred_instances.labels == 0) & (det_result.pred_instances.scores > 0.3)
        return det_result.pred_instances.bboxes[valid].cpu().numpy()

    async def estimate_pose(self, frame, bboxes: np.ndarray) -> PoseBatch:
        if bboxes is None or len(bboxes) == 0:
            return PoseBatch.empty()

        loop = asyncio.get_running_loop()
        pose_results = await loop.run_in_executor(
            self._executor,
            lambda: inference_topdown(self.pose_model, frame, bboxes, bbox_format="xyxy"),
        )
        data_samples = merge_data_samples(pose_results)
        return PoseBatch(
            keypoints=np.asarray(data_samples.pred_instances.keypoints, dtype=np.float32),
            keypoint_scores=np.asarray(data_samples.pred_instances.keypoint_scores, dtype=np.float32),
        )
