"""检测 / 姿态推理流程以及 WebSocket 推流逻辑。"""

import asyncio
import base64
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

try:
    import psutil
except Exception:
    psutil = None

try:
    import torch
except Exception:
    torch = None

from services.annotation_service import flatten_annotation_boxes
from services.box_identity import box_collision_token, parse_collision_token
from services.inference_backends import create_inference_backend, resolve_backend_name


def _parse_cuda_device_index(device_name: str) -> int:
    name = (device_name or "").strip().lower()
    if name.startswith("cuda:"):
        try:
            return int(name.split(":", 1)[1])
        except Exception:
            return 0
    return 0


def _collect_resource_debug_line(device_name: str) -> str:
    parts = []

    if psutil is not None:
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            parts.append(f"cpu={cpu_percent:.1f}%")
        except Exception:
            parts.append("cpu=n/a")

        try:
            proc_mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
            parts.append(f"proc_mem={proc_mem_mb:.1f}MB")
        except Exception:
            parts.append("proc_mem=n/a")

        try:
            sys_mem_percent = psutil.virtual_memory().percent
            parts.append(f"sys_mem={sys_mem_percent:.1f}%")
        except Exception:
            parts.append("sys_mem=n/a")
    else:
        parts.append("cpu=n/a(psutil_missing)")

    gpu_info_ready = False
    if torch is not None and torch.cuda.is_available():
        gpu_idx = _parse_cuda_device_index(device_name)
        try:
            alloc_mb = torch.cuda.memory_allocated(gpu_idx) / (1024 * 1024)
            reserved_mb = torch.cuda.memory_reserved(gpu_idx) / (1024 * 1024)
            parts.append(f"gpu_mem_alloc={alloc_mb:.1f}MB")
            parts.append(f"gpu_mem_reserved={reserved_mb:.1f}MB")
            gpu_info_ready = True
        except Exception:
            pass

        try:
            smi = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(gpu_idx),
                ],
                capture_output=True,
                text=True,
                timeout=0.4,
                check=False,
            )
            if smi.returncode == 0 and smi.stdout.strip():
                cols = [c.strip() for c in smi.stdout.strip().splitlines()[0].split(",")]
                if len(cols) >= 3:
                    parts.append(f"gpu_util={cols[0]}%")
                    parts.append(f"gpu_mem={cols[1]}/{cols[2]}MB")
                    gpu_info_ready = True
        except Exception:
            pass

    if not gpu_info_ready:
        parts.append("gpu=n/a")

    return " ".join(parts)


def _scale_polygon_points(points, sx: float, sy: float):
    out = []
    for pt in points:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            out.append([float(pt[0]) * sx, float(pt[1]) * sy])
    return out


def _polygon_max_extent(points) -> tuple[float, float]:
    max_x = 0.0
    max_y = 0.0
    for pt in points:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            max_x = max(max_x, float(pt[0]))
            max_y = max(max_y, float(pt[1]))
    return max_x, max_y


def _norm_polygon_valid(norm_pts) -> bool:
    if not isinstance(norm_pts, list) or len(norm_pts) < 3:
        return False
    for pt in norm_pts:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        x = float(pt[0])
        y = float(pt[1])
        if x < -0.01 or x > 1.01 or y < -0.01 or y > 1.01:
            return False
    return True


def _scale_polygon_to_frame(pts, norm_pts, ann_w, ann_h, target_w: int, target_h: int):
    tw = float(target_w)
    th = float(target_h)
    if _norm_polygon_valid(norm_pts):
        return _scale_polygon_points(norm_pts, tw, th)

    if not isinstance(pts, list) or len(pts) < 3:
        return []

    max_x, max_y = _polygon_max_extent(pts)
    if ann_w and ann_h and ann_w > 0 and ann_h > 0:
        sx = tw / float(ann_w) if max_x <= float(ann_w) * 1.05 else tw / max_x
        sy = th / float(ann_h) if max_y <= float(ann_h) * 1.05 else th / max_y
    elif max_x > 0 and max_y > 0:
        sx = tw / max_x
        sy = th / max_y
    else:
        sx = sy = 1.0
    return _scale_polygon_points(pts, sx, sy)


def _build_scaled_boxes(raw_boxes, ann_w: float | None, ann_h: float | None, target_w: int, target_h: int):
    scaled = []
    for box in raw_boxes:
        pts = box.get("video_polygon", [])
        norm_pts = box.get("video_polygon_norm", [])
        mapped_pts = _scale_polygon_to_frame(pts, norm_pts, ann_w, ann_h, target_w, target_h)

        if len(mapped_pts) < 3:
            continue

        new_box = dict(box)
        new_box["video_polygon"] = mapped_pts
        new_box["orig_contour"] = np.int32(mapped_pts).reshape((-1, 1, 2))
        scaled.append(new_box)

    return scaled


def _persist_live_collisions(
    app_config: dict,
    collisions: list,
    alarm_collisions: list,
    *,
    skeletons: list | None = None,
    infer_width: int = 0,
    infer_height: int = 0,
) -> None:
    """Headless 推理时将碰撞/骨架写入 status 文件，供监控页轮询。"""
    camera_id = os.environ.get("INFERENCE_CAMERA_ID", "").strip()
    if not camera_id:
        return
    status_dir = os.environ.get(
        "INFERENCE_STATUS_DIR",
        os.path.join(
            str(app_config.get("paths", {}).get("base_localdata_dir", "localdata")),
            "inference",
        ),
    )
    path = os.path.join(status_dir, f"{camera_id}.status.json")
    try:
        data: dict = {
            "camera_id": camera_id,
            "state": "running",
            "message": "",
            "updated_at": time.time(),
            "pid": os.getpid(),
        }
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                data.update(loaded)
        data["collisions"] = list(collisions)
        data["alarm_collisions"] = list(alarm_collisions)
        data["skeletons"] = list(skeletons) if skeletons is not None else data.get("skeletons", [])
        if infer_width > 0 and infer_height > 0:
            data["infer_width"] = int(infer_width)
            data["infer_height"] = int(infer_height)
        backend = os.environ.get("INFERENCE_BACKEND", "").strip()
        if backend:
            data["backend"] = backend
        data["updated_at"] = time.time()
        data["state"] = data.get("state") or "running"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except (OSError, json.JSONDecodeError):
        pass


def _compute_infer_resolution(source_w: int, source_h: int, target_height: int):
    target_h = max(120, int(target_height))
    if source_h <= target_h:
        infer_w = source_w
        infer_h = source_h
        resize_needed = False
    else:
        infer_h = target_h
        infer_w = int(round(source_w * (infer_h / float(source_h))))
        infer_w = max(2, infer_w - (infer_w % 2))
        infer_h = max(2, infer_h - (infer_h % 2))
        resize_needed = True
    return infer_w, infer_h, resize_needed


@dataclass
class TrackState:
    abs_x: float
    abs_y: float
    ts_sec: float


class PersonTrackAssigner:
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
    def __init__(self, app_config: dict, state, callback_reporter=None):
        self.app_config = app_config
        self.state = state
        self.callback_reporter = callback_reporter
        self._perception_backend = None
        self._background_task = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")

    def debug_visualization_enabled(self) -> bool:
        debug_cfg = self.app_config.get("debug-info", {})
        return isinstance(debug_cfg, dict) and bool(debug_cfg.get("enabled", False))

    def _backend(self):
        if self._perception_backend is None:
            backend_name = resolve_backend_name(self.app_config)
            print(f"ℹ️ 推理后端: {backend_name}")
            self._perception_backend = create_inference_backend(self.app_config, self._executor)
        return self._perception_backend

    def ensure_models_loaded(self):
        self._backend().ensure_loaded()

    async def _run_detection(self, frame):
        return await self._backend().detect_bboxes(frame)

    async def _run_pose(self, frame, bboxes):
        return await self._backend().estimate_pose(frame, bboxes)

    async def start_inference(self):
        self.ensure_models_loaded()
        if self.state.is_inferencing:
            return {"status": "success", "mode": "running"}

        self.state.is_inferencing = True

        if self.state.source_type == "stream":
            if self._background_task is None or self._background_task.done():
                self._background_task = asyncio.create_task(self.websocket_inference(_NullWebSocket()))
            return {"status": "success", "mode": "headless"}

        return {"status": "success", "mode": "visual"}

    async def websocket_inference(self, websocket: WebSocket):
        is_null_ws = isinstance(websocket, _NullWebSocket)
        visualization_enabled = self.debug_visualization_enabled() and (not is_null_ws)

        if (not is_null_ws) and self._background_task is not None and (not self._background_task.done()):
            self.state.is_inferencing = False
            try:
                await asyncio.wait_for(self._background_task, timeout=2.0)
            except Exception:
                self._background_task.cancel()
            finally:
                self._background_task = None
            self.state.is_inferencing = True

        if not visualization_enabled and (not is_null_ws):
            await websocket.accept()
            await websocket.send_json({
                "status": "debug-visual-disabled",
                "message": "debug-info.enabled=false，仅执行后台推理与回调，不推送前端可视化帧",
            })
            await websocket.close()
            return

        await websocket.accept()
        cap = None

        if self.state.source_type == "stream" and visualization_enabled and not self.state.is_inferencing:
            self.ensure_models_loaded()
            self.state.is_inferencing = True
            print("✅ 已在网络流模式自动启动推理会话")

        json_file_path = self.state.json_path or self.app_config["paths"]["default_json_file"]

        if not os.path.exists(json_file_path):
            print(f"⚠️ [警告] 无法启动推理：未找到配置文件 {json_file_path}，请先完成标注！")
            await websocket.close()
            return

        with open(json_file_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        raw_boxes = flatten_annotation_boxes(config_data)

        annotation_size = config_data.get("annotation_size", {}) if isinstance(config_data, dict) else {}
        if not isinstance(annotation_size, dict):
            annotation_size = {}

        ann_w = annotation_size.get("width")
        ann_h = annotation_size.get("height")
        try:
            ann_w = float(ann_w) if ann_w is not None else None
            ann_h = float(ann_h) if ann_h is not None else None
        except Exception:
            ann_w, ann_h = None, None

        source_info = config_data.get("source_info", {}) if isinstance(config_data, dict) else {}
        if not isinstance(source_info, dict):
            source_info = {}
        if self.state.source_type == "stream":
            marked_camera_url = str(source_info.get("camera_url", "") or "").strip()
            if marked_camera_url and marked_camera_url != (self.state.source_url or ""):
                print(
                    "⚠️ [警告] 当前流地址与标注来源摄像头不一致: "
                    f"stream={self.state.source_url} annotation_camera={marked_camera_url}"
                )

        cap = cv2.VideoCapture(self.state.video_path)
        if self.state.source_type == "stream":
            stream_buffer_size = int(self.app_config["inference"].get("stream_buffer_size", 1))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, stream_buffer_size))

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stream_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        stream_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if stream_w <= 0 or stream_h <= 0:
            print("⚠️ [警告] 无法读取当前视频分辨率，停止本次推理")
            self.state.is_inferencing = False
            await websocket.close()
            if cap is not None:
                cap.release()
            return

        infer_cfg = self.app_config.get("inference", {})
        infer_w, infer_h, resize_needed = _compute_infer_resolution(
            stream_w,
            stream_h,
            int(infer_cfg.get("height", 480) or 480),
        )

        boxes = _build_scaled_boxes(raw_boxes, ann_w, ann_h, infer_w, infer_h)

        frame_rate = float(infer_cfg.get("frame_rate", 15) or 15)
        frame_rate = max(1.0, frame_rate)
        frame_period_sec = 1.0 / frame_rate

        pose_frame_interval = int(infer_cfg.get("pose_frame_interval", 3) or 3)
        pose_frame_interval = max(1, pose_frame_interval)

        preview_max_width = int(infer_cfg.get("preview_max_width", 640) or 640)
        preview_jpeg_quality = int(infer_cfg.get("preview_jpeg_quality", 60) or 60)

        alarm_min_consecutive_frames = int(infer_cfg.get("alarm_min_consecutive_frames", 3))
        alarm_cooldown_frames = int(infer_cfg.get("alarm_cooldown_frames", 12))

        debug_cfg = self.app_config.get("debug-info", {})
        if not isinstance(debug_cfg, dict):
            debug_cfg = {}
        debug_enabled = bool(debug_cfg.get("enabled", False))
        try:
            debug_interval_frames = int(debug_cfg.get("interval_frames", 30))
        except Exception:
            debug_interval_frames = 30
        debug_interval_frames = max(1, debug_interval_frames)

        print(
            f"ℹ️ 推理参数: source={stream_w}x{stream_h} height={infer_h} "
            f"frame_rate={frame_rate} pose_frame_interval={pose_frame_interval} "
            f"resize={'on' if resize_needed else 'off'}"
        )

        start_time = time.time()
        frame_count = 0
        last_frame_started_at = time.monotonic()

        cached_bboxes = np.empty((0, 4), dtype=np.float32)
        cached_skeletons_data = []
        cached_collisions = []
        cached_alarm_collisions = []
        cached_report_event_ids = []
        box_consecutive_hits = {}
        box_last_alarm_frame = {}

        person_assigner = PersonTrackAssigner(max_match_dist=220.0, stale_sec=1.2)

        try:
            while cap.isOpened() and self.state.is_inferencing:
                loop_started_at = time.monotonic()

                ret, frame = await asyncio.get_running_loop().run_in_executor(self._executor, cap.read)
                if not ret:
                    print("✅ 视频推理完成，停止当前会话")
                    self.state.is_inferencing = False
                    break

                frame_count += 1
                if resize_needed:
                    raw_frame = cv2.resize(frame, (infer_w, infer_h), interpolation=cv2.INTER_AREA)
                else:
                    raw_frame = frame

                cached_bboxes = await self._run_detection(raw_frame)
                run_pose = ((frame_count - 1) % pose_frame_interval == 0)

                if run_pose:
                    active_collisions = []
                    alarm_collisions = []
                    skeletons_data = []
                    report_event_ids = []

                    if len(cached_bboxes) > 0:
                        pose_batch = await self._run_pose(raw_frame, cached_bboxes)
                        kpts_all = pose_batch.keypoints
                        scores_all = pose_batch.keypoint_scores
                        used_person_track_ids = set()

                        for p_idx in range(pose_batch.num_persons):
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
                                person_pts.append([
                                    float(kpts_all[p_idx][k][0]),
                                    float(kpts_all[p_idx][k][1]),
                                    float(scores_all[p_idx][k]),
                                ])
                            skeletons_data.append({
                                "person_id": p_idx,
                                "person_track_id": person_track_id,
                                "keypoints": person_pts,
                            })

                            for _, kpt_idx in [("左手", 9), ("右手", 10)]:
                                wrist = kpts_all[p_idx][kpt_idx]
                                score = scores_all[p_idx][kpt_idx]
                                if score > 0.3:
                                    for box in boxes:
                                        if cv2.pointPolygonTest(
                                            box["orig_contour"],
                                            (float(wrist[0]), float(wrist[1])),
                                            False,
                                        ) >= 0:
                                            token = box_collision_token(box)
                                            if token:
                                                active_collisions.append(token)
                                            break

                    cached_skeletons_data = skeletons_data
                    cached_collisions = list(set(active_collisions))

                    current_collision_tokens = set(cached_collisions)

                    for token in list(box_consecutive_hits.keys()):
                        if token not in current_collision_tokens:
                            box_consecutive_hits[token] = 0

                    for token in current_collision_tokens:
                        box_consecutive_hits[token] = box_consecutive_hits.get(token, 0) + 1
                        last_alarm_frame = box_last_alarm_frame.get(token, -10**9)
                        if (
                            box_consecutive_hits[token] >= alarm_min_consecutive_frames
                            and frame_count - last_alarm_frame >= alarm_cooldown_frames
                        ):
                            alarm_collisions.append(token)
                            box_last_alarm_frame[token] = frame_count

                    cached_alarm_collisions = alarm_collisions

                    if self.callback_reporter is not None and cached_alarm_collisions:
                        upload_tag = self.state.upload_tag or f"u{int(self.state.upload_id or 0):06d}"
                        video_time_sec = frame_count / video_fps
                        for collision in cached_alarm_collisions:
                            shelf_code, box_id = parse_collision_token(collision)
                            if not box_id:
                                continue
                            event_id = self.callback_reporter.enqueue_pick_finished(
                                box_id=box_id,
                                frame_idx=frame_count,
                                video_time_sec=video_time_sec,
                                upload_tag=upload_tag,
                                shelf_code=shelf_code or None,
                            )
                            if event_id is not None:
                                report_event_ids.append(event_id)

                    cached_report_event_ids = report_event_ids

                skeletons_data = cached_skeletons_data
                active_collisions = cached_collisions
                alarm_collisions = cached_alarm_collisions
                report_event_ids = cached_report_event_ids

                if is_null_ws:
                    _persist_live_collisions(
                        self.app_config,
                        active_collisions,
                        alarm_collisions,
                        skeletons=skeletons_data,
                        infer_width=infer_w,
                        infer_height=infer_h,
                    )

                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0

                video_time_sec = frame_count / video_fps
                m, s = int(video_time_sec // 60), int(video_time_sec % 60)
                ms = int((video_time_sec - int(video_time_sec)) * 1000)
                formatted_time = f"{m:02d}:{s:02d}.{ms:03d}"

                if debug_enabled and (frame_count % debug_interval_frames == 0):
                    resource_line = _collect_resource_debug_line(self.app_config["models"].get("device", ""))
                    print(
                        f"[DEBUG-INFO] frame={frame_count} fps={round(current_fps, 1)} "
                        f"video_time={formatted_time} {resource_line}"
                    )

                if visualization_enabled:
                    target_w = min(preview_max_width, infer_w)
                    if target_w < infer_w:
                        target_h = int(infer_h * (target_w / infer_w))
                        frame_for_encode = cv2.resize(raw_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    else:
                        frame_for_encode = raw_frame

                    _, buffer = cv2.imencode(
                        ".jpg",
                        frame_for_encode,
                        [int(cv2.IMWRITE_JPEG_QUALITY), preview_jpeg_quality],
                    )
                    b64_str = base64.b64encode(buffer).decode("utf-8")

                    payload = {
                        "image": b64_str,
                        "orig_width": infer_w,
                        "skeletons": skeletons_data,
                        "collisions": active_collisions,
                        "alarm_collisions": alarm_collisions,
                        "callback_event_ids": report_event_ids,
                        "annotation_source": source_info,
                        "stats": {
                            "fps": round(current_fps, 1),
                            "video_time": formatted_time,
                            "frame_idx": frame_count,
                        },
                    }

                    await websocket.send_json(payload)

                elapsed_loop = time.monotonic() - loop_started_at
                sleep_sec = frame_period_sec - elapsed_loop
                if sleep_sec > 0:
                    await asyncio.sleep(sleep_sec)

        except WebSocketDisconnect:
            if not is_null_ws:
                print("前端连接断开")
        finally:
            self.state.is_inferencing = False
            if cap is not None:
                cap.release()

            if (not is_null_ws) and self.state.source_type == "stream":
                self.state.is_inferencing = True
                if self._background_task is None or self._background_task.done():
                    self._background_task = asyncio.create_task(self.websocket_inference(_NullWebSocket()))


class _NullWebSocket:
    async def accept(self):
        return

    async def send_json(self, _payload):
        return

    async def close(self):
        return
