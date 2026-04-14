"""视频上传、转码和首帧提取相关函数。

这个模块只负责和视频文件打交道，把上传文件落盘、转码、提取首帧
这些 I/O 逻辑与推理流程分开。
"""

import base64
import os
import shutil
from typing import Tuple

import cv2
from fastapi import UploadFile

from core.state import STATE


def reserve_next_upload_id(counter_file: str, base_localdata_dir: str) -> int:
    """预留一个持久化递增的上传编号。"""
    os.makedirs(base_localdata_dir, exist_ok=True)
    current = 0
    if os.path.exists(counter_file):
        try:
            with open(counter_file, "r", encoding="utf-8") as f:
                current = int((f.read() or "0").strip())
        except Exception:
            current = 0

    next_id = current + 1
    with open(counter_file, "w", encoding="utf-8") as f:
        f.write(str(next_id))
    return next_id


def transcode_video_to_480p(src_path: str, dst_path: str) -> Tuple[int, int]:
    """将上传视频转码到最高 480p，并保持宽高比不变。"""
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


async def handle_video_upload(file: UploadFile, upload_dir: str, upload_480p_dir: str, json_dir: str, counter_file: str, base_localdata_dir: str):
    """保存新上传的视频，并初始化会话路径到 STATE。"""
    os.makedirs(upload_dir, exist_ok=True)
    os.makedirs(upload_480p_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    upload_id = reserve_next_upload_id(counter_file, base_localdata_dir)
    upload_tag = f"u{upload_id:06d}"

    raw_video_path = os.path.join(upload_dir, f"{upload_tag}_src.mp4")
    compressed_video_path = os.path.join(upload_480p_dir, f"{upload_tag}_480p.mp4")
    upload_json_path = os.path.join(json_dir, f"{upload_tag}_boxes.json")

    with open(raw_video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        out_w, out_h = transcode_video_to_480p(raw_video_path, compressed_video_path)
        STATE.video_path = compressed_video_path
        print(f"✅ 上传视频已转码: {compressed_video_path} ({out_w}x{out_h})")
    except Exception as e:
        STATE.video_path = raw_video_path
        print(f"⚠️ 480p 转码失败，回退原视频: {e}")

    STATE.is_inferencing = False
    STATE.upload_id = upload_id
    STATE.upload_tag = upload_tag
    STATE.json_path = upload_json_path

    return {"status": "success", "upload_id": upload_id, "upload_tag": upload_tag}


def get_first_frame_b64(video_path: str):
    """读取视频第一帧，并返回 base64 编码的 JPEG 数据。"""
    if not video_path or not os.path.exists(video_path):
        return {"error": "No video"}

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return {"image": base64.b64encode(buffer).decode('utf-8')}
    return {"error": "Read failed"}
