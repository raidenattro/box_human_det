import cv2
import numpy as np
import json
import base64
import os
import shutil
import asyncio
import time
import math
import csv
import uuid
import hashlib
from collections import deque
from dataclasses import dataclass
from typing import Tuple
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn

# ==========================================
# 🚀 MMPose & MMDet 导入
# ==========================================
from mmdet.apis import init_detector, inference_detector
from mmpose.apis import init_model as init_pose_model, inference_topdown
from mmpose.structures import merge_data_samples
from mmpose.utils import register_all_modules, adapt_mmdet_pipeline
register_all_modules()

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@dataclass
class TrackState:
    abs_x: float
    abs_y: float
    ts_sec: float


class WristTrackAssigner:
    """Assign a stable hand track id with nearest-neighbor matching in pixel space."""

    def __init__(self, max_match_dist=150.0, stale_sec=1.0):
        self.max_match_dist = max_match_dist
        self.stale_sec = stale_sec
        self.next_id = 1
        self.tracks = {}

    def _cleanup(self, now_ts: float):
        dead_keys = [k for k, st in self.tracks.items() if now_ts - st.ts_sec > self.stale_sec]
        for k in dead_keys:
            self.tracks.pop(k, None)

    def assign(self, person_id_raw: int, hand_side: str, abs_x: float, abs_y: float, now_ts: float) -> int:
        self._cleanup(now_ts)
        best_key = None
        best_dist = 1e9

        for (pid, side, tid), st in self.tracks.items():
            if pid != person_id_raw:
                continue
            if side != hand_side:
                continue
            dist = math.hypot(abs_x - st.abs_x, abs_y - st.abs_y)
            if dist < best_dist:
                best_dist = dist
                best_key = (pid, side, tid)

        if best_key is None or best_dist > self.max_match_dist:
            tid = self.next_id
            self.next_id += 1
            self.tracks[(person_id_raw, hand_side, tid)] = TrackState(abs_x=abs_x, abs_y=abs_y, ts_sec=now_ts)
            return tid

        _, _, tid = best_key
        self.tracks[(person_id_raw, hand_side, tid)] = TrackState(abs_x=abs_x, abs_y=abs_y, ts_sec=now_ts)
        return tid


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


class ActionFeatureExtractorV2:
    """Collect per-frame hand interaction features for action model training."""

    def __init__(self, csv_filename="localdata/action_dataset.csv", auto_flush_every=200):
        self.history = {}
        self.csv_filename = csv_filename
        self.auto_flush_every = auto_flush_every
        self.write_count = 0

        os.makedirs(os.path.dirname(csv_filename) or ".", exist_ok=True)
        self.csv_file = open(self.csv_filename, mode='a', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)

        if os.path.getsize(self.csv_filename) == 0:
            headers = [
                "session_id", "video_id", "ts_sec", "frame_idx",
                "track_id", "person_id_raw", "person_track_id", "hand_side",
                "norm_x", "norm_y", "score",
                "v_x", "v_y", "a_x", "a_y",
                "is_in_box", "dist_to_box_center", "box_id"
            ]
            self.csv_writer.writerow(headers)

    def _save_row(self, row):
        self.csv_writer.writerow(row)
        self.write_count += 1
        if self.write_count % self.auto_flush_every == 0:
            self.csv_file.flush()

    def close(self):
        try:
            self.csv_file.flush()
            self.csv_file.close()
        except Exception:
            pass

    def extract_and_save(
        self,
        session_id,
        video_id,
        ts_sec,
        frame_idx,
        person_id_raw,
        person_track_id,
        hand_assigner,
        kpts,
        scores,
        boxes,
    ):
        if len(scores) <= 10:
            return

        # Build body-centric coordinates with neck as origin and shoulder width as scale.
        if scores[5] < 0.2 or scores[6] < 0.2:
            return

        neck_x = (kpts[5][0] + kpts[6][0]) / 2.0
        neck_y = (kpts[5][1] + kpts[6][1]) / 2.0
        shoulder_width = max(math.hypot(kpts[5][0] - kpts[6][0], kpts[5][1] - kpts[6][1]), 10.0)

        for hand_side, wrist_idx in [("left", 9), ("right", 10)]:
            score = float(scores[wrist_idx])
            if score < 0.15:
                continue

            wrist_x = float(kpts[wrist_idx][0])
            wrist_y = float(kpts[wrist_idx][1])
            norm_x = (wrist_x - neck_x) / shoulder_width
            norm_y = (wrist_y - neck_y) / shoulder_width

            track_id = hand_assigner.assign(person_track_id, hand_side, wrist_x, wrist_y, ts_sec)
            hand_key = f"{session_id}_{person_track_id}_{track_id}_{hand_side}"

            if hand_key not in self.history:
                self.history[hand_key] = deque(maxlen=3)

            history = self.history[hand_key]
            history.append((norm_x, norm_y, ts_sec))

            v_x, v_y, a_x, a_y = 0.0, 0.0, 0.0, 0.0
            if len(history) >= 2:
                x1, y1, t1 = history[-2]
                x2, y2, t2 = history[-1]
                dt = max(t2 - t1, 1e-3)
                v_x = (x2 - x1) / dt
                v_y = (y2 - y1) / dt

            if len(history) >= 3:
                x0, y0, t0 = history[-3]
                x1, y1, t1 = history[-2]
                x2, y2, t2 = history[-1]
                dt1 = max(t1 - t0, 1e-3)
                dt2 = max(t2 - t1, 1e-3)
                v1_x = (x1 - x0) / dt1
                v1_y = (y1 - y0) / dt1
                v2_x = (x2 - x1) / dt2
                v2_y = (y2 - y1) / dt2
                a_dt = max(t2 - t0, 1e-3)
                a_x = (v2_x - v1_x) / a_dt
                a_y = (v2_y - v1_y) / a_dt

            is_in_box = 0
            target_box_id = -1
            min_dist = float('inf')

            for box in boxes:
                contour = box['orig_contour']
                moments = cv2.moments(contour)
                if moments["m00"] != 0:
                    cx = float(moments["m10"] / moments["m00"])
                    cy = float(moments["m01"] / moments["m00"])
                else:
                    cx, cy = float(contour[0][0][0]), float(contour[0][0][1])

                dist_to_center = math.hypot(wrist_x - cx, wrist_y - cy) / shoulder_width

                if cv2.pointPolygonTest(contour, (wrist_x, wrist_y), False) >= 0:
                    is_in_box = 1
                    target_box_id = box['box_id']
                    min_dist = dist_to_center
                    break

                if dist_to_center < min_dist:
                    min_dist = dist_to_center
                    target_box_id = box['box_id']

            row = [
                session_id,
                video_id,
                round(ts_sec, 4),
                frame_idx,
                track_id,
                person_id_raw,
                person_track_id,
                hand_side,
                round(norm_x, 4),
                round(norm_y, 4),
                round(score, 4),
                round(v_x, 4),
                round(v_y, 4),
                round(a_x, 4),
                round(a_y, 4),
                is_in_box,
                round(min_dist, 4),
                target_box_id,
            ]
            self._save_row(row)


def transcode_video_to_480p(src_path: str, dst_path: str) -> Tuple[int, int]:
    """Transcode uploaded video to max 480p (keep aspect ratio, no upscale)."""
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开上传视频进行转码")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    target_h = min(480, src_h)
    target_w = int(round(src_w * (target_h / src_h)))
    # 保证编码尺寸为偶数，避免某些编码器报错
    target_w = max(2, target_w - (target_w % 2))
    target_h = max(2, target_h - (target_h % 2))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dst_path, fourcc, fps, (target_w, target_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("无法创建 480p 转码输出文件")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            writer.write(resized)
    finally:
        cap.release()
        writer.release()

    return target_w, target_h

# ==========================================
# 🎯 全局状态与模型
# ==========================================
APP_STATE = {
    "video_path": "",
    "is_inferencing": False
}
BASE_LOCALDATA_DIR = "localdata"
UPLOAD_DIR = os.path.join(BASE_LOCALDATA_DIR, "upload")
UPLOAD_480P_DIR = os.path.join(UPLOAD_DIR, "480p")
JSON_DIR = os.path.join(BASE_LOCALDATA_DIR, "json")
CSV_DIR = os.path.join(BASE_LOCALDATA_DIR, "csv")
JSON_FILE = os.path.join(JSON_DIR, "precise_boxes_new.json")
COUNTER_FILE = os.path.join(BASE_LOCALDATA_DIR, "upload_counter.txt")


def reserve_next_upload_id() -> int:
    """Reserve a monotonically increasing upload id persisted on disk."""
    os.makedirs(BASE_LOCALDATA_DIR, exist_ok=True)
    current = 0
    if os.path.exists(COUNTER_FILE):
        try:
            with open(COUNTER_FILE, "r", encoding="utf-8") as f:
                current = int((f.read() or "0").strip())
        except Exception:
            current = 0

    next_id = current + 1
    with open(COUNTER_FILE, "w", encoding="utf-8") as f:
        f.write(str(next_id))
    return next_id

det_model = None
pose_model = None

# ==========================================
# 🌐 基础 HTTP 接口
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def read_root():
    # 直接读取并返回 HTML，不再依赖 request 传参
    with open("templates/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(UPLOAD_480P_DIR, exist_ok=True)
    os.makedirs(JSON_DIR, exist_ok=True)
    os.makedirs(CSV_DIR, exist_ok=True)

    upload_id = reserve_next_upload_id()
    upload_tag = f"u{upload_id:06d}"

    raw_video_path = os.path.join(UPLOAD_DIR, f"{upload_tag}_src.mp4")
    compressed_video_path = os.path.join(UPLOAD_480P_DIR, f"{upload_tag}_480p.mp4")
    upload_json_path = os.path.join(JSON_DIR, f"{upload_tag}_boxes.json")

    with open(raw_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        out_w, out_h = transcode_video_to_480p(raw_video_path, compressed_video_path)
        APP_STATE["video_path"] = compressed_video_path
        APP_STATE["upload_id"] = upload_id
        APP_STATE["upload_tag"] = upload_tag
        APP_STATE["json_path"] = upload_json_path
        print(f"✅ 上传视频已转码: {compressed_video_path} ({out_w}x{out_h})")
    except Exception as e:
        # 转码失败时保底回退到原视频，避免阻断流程
        APP_STATE["video_path"] = raw_video_path
        APP_STATE["upload_id"] = upload_id
        APP_STATE["upload_tag"] = upload_tag
        APP_STATE["json_path"] = upload_json_path
        print(f"⚠️ 480p 转码失败，回退原视频: {e}")

    APP_STATE["is_inferencing"] = False
    return {"status": "success", "upload_id": upload_id, "upload_tag": upload_tag}

@app.get("/get_first_frame")
async def get_first_frame():
    """专门给前端标注用的接口，提取第一帧原图"""
    if not APP_STATE["video_path"] or not os.path.exists(APP_STATE["video_path"]):
        return {"error": "No video"}
    cap = cv2.VideoCapture(APP_STATE["video_path"])
    ret, frame = cap.read()
    cap.release()
    if ret:
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return {"image": base64.b64encode(buffer).decode('utf-8')}
    return {"error": "Read failed"}

@app.post("/save_annotation")
async def save_annotation(data: dict):
    """前端自己算好网格后，把 JSON 丢给后端保存"""
    os.makedirs(JSON_DIR, exist_ok=True)
    json_path = APP_STATE.get("json_path", JSON_FILE)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    return {"status": "success", "json_path": json_path}

@app.post("/start_inference")
async def start_inference():
    global det_model, pose_model
    if det_model is None:
        print("🚀 正在加载 AI 模型...")
        device = 'cuda:0' # ⚠️ 请确保这里符合你的服务器配置
        det_model = init_detector('/home/zyqiao/workspace/mmpose/demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py',
                                  'https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth', device=device)
        det_model.cfg = adapt_mmdet_pipeline(det_model.cfg)
        pose_model = init_pose_model('/home/zyqiao/workspace/mmpose/configs/body_2d_keypoint/rtmpose/body8/rtmpose-m_8xb256-420e_body8-256x192.py',
                                     'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.pth', device=device,
                                     cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=False))))
    
    APP_STATE["is_inferencing"] = True
    return {"status": "success"}

# ==========================================
# 🧠 核心：纯数据 WebSocket 推流引擎
# ==========================================
@app.websocket("/ws/inference")
async def websocket_inference(websocket: WebSocket):
    await websocket.accept()
    feature_extractor = None
    cap = None

    json_file_path = APP_STATE.get("json_path", JSON_FILE)
    
    if not os.path.exists(json_file_path):
        print(f"⚠️ [警告] 无法启动推理：未找到配置文件 {json_file_path}，请先完成前端标注！")
        await websocket.close()
        return
        
    with open(json_file_path, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    boxes = config_data['boxes']
    for box in boxes:
        box['orig_contour'] = np.int32(box['video_polygon']).reshape((-1, 1, 2))

    cap = cv2.VideoCapture(APP_STATE["video_path"])
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_time = time.time()
    frame_count = 0

    # 轻量时序优化：检测每 3 帧执行一次，姿态每 2 帧执行一次。
    det_interval = 3
    pose_interval = 2
    cached_bboxes = np.empty((0, 4), dtype=np.float32)
    cached_skeletons_data = []
    cached_collisions = []

    session_id = str(uuid.uuid4())[:8]
    upload_id = int(APP_STATE.get("upload_id", 0))
    upload_tag = APP_STATE.get("upload_tag", f"u{upload_id:06d}" if upload_id > 0 else "u000000")
    video_src = APP_STATE.get("video_path", "")
    if os.path.exists(video_src):
        f_stat = os.stat(video_src)
        unique_file_str = f"{f_stat.st_size}_{f_stat.st_mtime}"
        video_id = hashlib.md5(unique_file_str.encode("utf-8")).hexdigest()[:10]
    else:
        video_id = "unknown"

    os.makedirs(CSV_DIR, exist_ok=True)
    csv_name = os.path.join(CSV_DIR, f"action_dataset_{upload_tag}_s{session_id}.csv")
    feature_extractor = ActionFeatureExtractorV2(csv_filename=csv_name)
    person_assigner = PersonTrackAssigner(max_match_dist=220.0, stale_sec=1.2)
    hand_assigner = WristTrackAssigner(max_match_dist=150.0, stale_sec=1.0)
    
    try:
        while cap.isOpened() and APP_STATE["is_inferencing"]:
            ret, frame = cap.read()
            if not ret: 
                print("✅ 视频推理完成，停止当前会话")
                APP_STATE["is_inferencing"] = False
                break
                
            frame_count += 1
            raw_frame = frame
            run_det = ((frame_count - 1) % det_interval == 0)
            run_pose = ((frame_count - 1) % pose_interval == 0)

            if run_det:
                det_result = inference_detector(det_model, raw_frame)
                valid = (det_result.pred_instances.labels == 0) & (det_result.pred_instances.scores > 0.3)
                cached_bboxes = det_result.pred_instances.bboxes[valid].cpu().numpy()

            if run_pose:
                active_collisions = []
                skeletons_data = []

                if len(cached_bboxes) > 0:
                    pose_results = inference_topdown(pose_model, raw_frame, cached_bboxes, bbox_format='xyxy')
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

                        feature_extractor.extract_and_save(
                            session_id=session_id,
                            video_id=video_id,
                            ts_sec=(frame_count / video_fps),
                            frame_idx=frame_count,
                            person_id_raw=p_idx,
                            person_track_id=person_track_id,
                            hand_assigner=hand_assigner,
                            kpts=kpts_all[p_idx],
                            scores=scores_all[p_idx],
                            boxes=boxes,
                        )

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
            else:
                skeletons_data = cached_skeletons_data
                active_collisions = cached_collisions
            
            # 仅在原图宽度大于 640 时降采样，避免无意义放大带来的额外开销。
            target_w = min(640, orig_w)
            if target_w < orig_w:
                target_h = int(orig_h * (target_w / orig_w))
                frame_for_encode = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            else:
                frame_for_encode = frame

            _, buffer = cv2.imencode('.jpg', frame_for_encode, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            b64_str = base64.b64encode(buffer).decode('utf-8')
            
            # 🌟 3. 计算实时统计信息
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            
            video_time_sec = frame_count / video_fps
            m, s = int(video_time_sec // 60), int(video_time_sec % 60)
            ms = int((video_time_sec - int(video_time_sec)) * 1000)
            formatted_time = f"{m:02d}:{s:02d}.{ms:03d}"
            
            # 4. 构建数据载荷
            payload = {
                "image": b64_str,
                "orig_width": orig_w, # 告诉前端原图多大，以便前端坐标换算
                "skeletons": skeletons_data,
                "collisions": active_collisions,
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
        if feature_extractor is not None:
            feature_extractor.close()
        cap.release()
        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8045)