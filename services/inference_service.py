"""检测 / 姿态推理流程以及 WebSocket 推流逻辑。

这个模块有意把最重的运行逻辑集中在一起，保证入口文件保持轻量，
同时也方便后续只看一个文件就能找到完整推理链路。
"""

import asyncio
import base64
import json
import math
import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from core.state import STATE

from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model as init_pose_model, inference_topdown
from mmpose.structures import merge_data_samples
from mmpose.utils import register_all_modules, adapt_mmdet_pipeline

register_all_modules()


@dataclass
class TrackState:
    """最近邻跟踪器使用的单次轨迹快照。"""
    abs_x: float
    abs_y: float
    ts_sec: float


class PersonTrackAssigner:
    """Assign stable person ids across frames with nearest-neighbor matching."""

    def __init__(self, max_match_dist=220.0, stale_sec=1.2):
        self.max_match_dist = max_match_dist
        self.stale_sec = stale_sec
        self.next_id = 1
        self.tracks = {}

    def _cleanup(self, now_ts: float):
        dead_keys = [k for k, st in self.tracks.items() if now_ts - st.ts_sec > self.stale_sec]
        for k in dead_keys:
            self.tracks.pop(k, None)

    def assign(self, abs_x: float, abs_y: float, now_ts: float, occupied_track_ids=None) -> int:
        self._cleanup(now_ts)
        occupied = occupied_track_ids if occupied_track_ids is not None else set()

        best_tid = None
        best_dist = 1e9
        for tid, st in self.tracks.items():
            if tid in occupied:
                continue
            dist = math.hypot(abs_x - st.abs_x, abs_y - st.abs_y)
            if dist < best_dist:
                best_dist = dist
                best_tid = tid

        if best_tid is None or best_dist > self.max_match_dist:
            tid = self.next_id
            self.next_id += 1
            self.tracks[tid] = TrackState(abs_x=abs_x, abs_y=abs_y, ts_sec=now_ts)
            occupied.add(tid)
            return tid

        self.tracks[best_tid] = TrackState(abs_x=abs_x, abs_y=abs_y, ts_sec=now_ts)
        occupied.add(best_tid)
        return best_tid


class InferenceService:
    """管理模型生命周期和单视频的推理流。"""

    def __init__(self, app_config: dict, state, callback_reporter=None):
        """保存已解析配置和共享运行态。"""
        self.app_config = app_config
        self.state = state
        self.callback_reporter = callback_reporter
        self.det_model = None
        self.pose_model = None

    def ensure_models_loaded(self):
        """按需懒加载 MMDetection 和 MMPose 模型，每个进程只加载一次。"""
        if self.det_model is not None and self.pose_model is not None:
            return

        print("🚀 正在加载 AI 模型...")
        device = self.app_config["models"]["device"]
        self.det_model = init_detector(
            self.app_config["models"]["det_config"],
            self.app_config["models"]["det_checkpoint"],
            device=device,
        )
        self.det_model.cfg = adapt_mmdet_pipeline(self.det_model.cfg)
        self.pose_model = init_pose_model(
            self.app_config["models"]["pose_config"],
            self.app_config["models"]["pose_checkpoint"],
            device=device,
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))),
        )

    async def start_inference(self):
        """准备模型，并把共享状态切换为推理模式。"""
        self.ensure_models_loaded()
        self.state.is_inferencing = True
        return {"status": "success"}

    async def websocket_inference(self, websocket: WebSocket):
        """通过 WebSocket 向前端推送视频帧、骨架和碰撞信息。"""
        await websocket.accept()
        cap = None

        json_file_path = self.state.json_path or self.app_config["paths"]["default_json_file"]

        if not os.path.exists(json_file_path):
            print(f"⚠️ [警告] 无法启动推理：未找到配置文件 {json_file_path}，请先完成前端标注！")
            await websocket.close()
            return

        with open(json_file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        boxes = config_data['boxes']
        for box in boxes:
            box['orig_contour'] = np.int32(box['video_polygon']).reshape((-1, 1, 2))

        cap = cv2.VideoCapture(self.state.video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        start_time = time.time()
        frame_count = 0

        det_interval = int(self.app_config["inference"].get("det_interval", 3))
        pose_interval = int(self.app_config["inference"].get("pose_interval", 2))
        cached_bboxes = np.empty((0, 4), dtype=np.float32)
        cached_skeletons_data = []
        cached_collisions = []
        cached_report_event_ids = []

        person_assigner = PersonTrackAssigner(max_match_dist=220.0, stale_sec=1.2)

        try:
            while cap.isOpened() and self.state.is_inferencing:
                ret, frame = cap.read()
                if not ret:
                    print("✅ 视频推理完成，停止当前会话")
                    self.state.is_inferencing = False
                    break

                frame_count += 1
                raw_frame = frame
                run_det = ((frame_count - 1) % det_interval == 0)
                run_pose = ((frame_count - 1) % pose_interval == 0)

                if run_det:
                    det_result = inference_detector(self.det_model, raw_frame)
                    valid = (det_result.pred_instances.labels == 0) & (det_result.pred_instances.scores > 0.3)
                    cached_bboxes = det_result.pred_instances.bboxes[valid].cpu().numpy()

                if run_pose:
                    active_collisions = []
                    skeletons_data = []
                    report_event_ids = []

                    if len(cached_bboxes) > 0:
                        pose_results = inference_topdown(self.pose_model, raw_frame, cached_bboxes, bbox_format='xyxy')
                        data_samples = merge_data_samples(pose_results)

                        kpts_all = np.asarray(data_samples.pred_instances.keypoints)
                        scores_all = np.asarray(data_samples.pred_instances.keypoint_scores)
                        used_person_track_ids = set()

                        for p_idx in range(kpts_all.shape[0]):
                            if scores_all[p_idx][5] > 0.2 and scores_all[p_idx][6] > 0.2:
                                anchor_x = float((kpts_all[p_idx][5][0] + kpts_all[p_idx][6][0]) / 2.0)
                                anchor_y = float((kpts_all[p_idx][5][1] + kpts_all[p_idx][6][1]) / 2.0)
                            else:
                                x1, y1, x2, y2 = cached_bboxes[p_idx]
                                anchor_x = float((x1 + x2) / 2.0)
                                anchor_y = float((y1 + y2) / 2.0)

                            person_track_id = person_assigner.assign(
                                anchor_x,
                                anchor_y,
                                now_ts=(frame_count / video_fps),
                                occupied_track_ids=used_person_track_ids,
                            )

                            person_pts = []
                            for k in range(kpts_all.shape[1]):
                                person_pts.append([float(kpts_all[p_idx][k][0]), float(kpts_all[p_idx][k][1]), float(scores_all[p_idx][k])])
                            skeletons_data.append({"person_id": p_idx, "person_track_id": person_track_id, "keypoints": person_pts})

                            for _, kpt_idx in [("左手", 9), ("右手", 10)]:
                                wrist = kpts_all[p_idx][kpt_idx]
                                score = scores_all[p_idx][kpt_idx]
                                if score > 0.3:
                                    for box in boxes:
                                        if cv2.pointPolygonTest(box['orig_contour'], (float(wrist[0]), float(wrist[1])), False) >= 0:
                                            active_collisions.append(f"Box_{box['box_id']}")
                                            break

                    cached_skeletons_data = skeletons_data
                    cached_collisions = list(set(active_collisions))

                    # 将碰撞事件异步入队给回调上报服务，避免阻塞推理主循环。
                    if self.callback_reporter is not None and cached_collisions:
                        upload_tag = self.state.upload_tag or f"u{int(self.state.upload_id or 0):06d}"
                        video_time_sec = frame_count / video_fps
                        for collision in cached_collisions:
                            if not collision.startswith("Box_"):
                                continue
                            try:
                                box_id = int(collision.split("_", 1)[1])
                            except Exception:
                                continue
                            event_id = self.callback_reporter.enqueue_pick_finished(
                                box_id=box_id,
                                frame_idx=frame_count,
                                video_time_sec=video_time_sec,
                                upload_tag=upload_tag,
                            )
                            if event_id is not None:
                                report_event_ids.append(event_id)

                    cached_report_event_ids = report_event_ids
                else:
                    skeletons_data = cached_skeletons_data
                    active_collisions = cached_collisions
                    report_event_ids = cached_report_event_ids

                target_w = min(640, orig_w)
                if target_w < orig_w:
                    target_h = int(orig_h * (target_w / orig_w))
                    frame_for_encode = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                else:
                    frame_for_encode = frame

                _, buffer = cv2.imencode('.jpg', frame_for_encode, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                b64_str = base64.b64encode(buffer).decode('utf-8')

                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0

                video_time_sec = frame_count / video_fps
                m, s = int(video_time_sec // 60), int(video_time_sec % 60)
                ms = int((video_time_sec - int(video_time_sec)) * 1000)
                formatted_time = f"{m:02d}:{s:02d}.{ms:03d}"

                payload = {
                    "image": b64_str,
                    "orig_width": orig_w,
                    "skeletons": skeletons_data,
                    "collisions": active_collisions,
                    "callback_event_ids": report_event_ids,
                    "stats": {
                        "fps": round(current_fps, 1),
                        "video_time": formatted_time,
                        "frame_idx": frame_count
                    }
                }

                await websocket.send_json(payload)
                await asyncio.sleep(0.001)

        except WebSocketDisconnect:
            print("前端连接断开")
        finally:
            if cap is not None:
                cap.release()
