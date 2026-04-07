import cv2
import numpy as np
import json
import base64
import os
import shutil
import asyncio
import time
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
JSON_FILE = "precise_boxes_new.json"

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
    os.makedirs("localdata", exist_ok=True)
    raw_video_path = "localdata/current_demo_src.mp4"
    compressed_video_path = "localdata/current_demo_480p.mp4"

    with open(raw_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        out_w, out_h = transcode_video_to_480p(raw_video_path, compressed_video_path)
        APP_STATE["video_path"] = compressed_video_path
        print(f"✅ 上传视频已转码: {compressed_video_path} ({out_w}x{out_h})")
    except Exception as e:
        # 转码失败时保底回退到原视频，避免阻断流程
        APP_STATE["video_path"] = raw_video_path
        print(f"⚠️ 480p 转码失败，回退原视频: {e}")

    APP_STATE["is_inferencing"] = False
    return {"status": "success"}

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
    with open(JSON_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    return {"status": "success"}

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
    
    if not os.path.exists(JSON_FILE):
        await websocket.close()
        return
        
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
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
    
    try:
        while cap.isOpened() and APP_STATE["is_inferencing"]:
            ret, frame = cap.read()
            if not ret: 
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
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

                    for p_idx in range(kpts_all.shape[0]):
                        person_pts = []
                        for k in range(kpts_all.shape[1]):
                            person_pts.append([float(kpts_all[p_idx][k][0]), float(kpts_all[p_idx][k][1]), float(scores_all[p_idx][k])])
                        skeletons_data.append({"person_id": p_idx, "keypoints": person_pts})

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
        cap.release()
        
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8045)