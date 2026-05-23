"""摄像头地址管理与抓帧服务。"""

import base64
import json
import os
from typing import List

import cv2

from services.video_service import read_non_black_frame, resize_frame_to_height


def load_camera_ips(camera_ips_file: str) -> List[dict]:
    if not os.path.exists(camera_ips_file):
        return []

    try:
        data = json.loads(open(camera_ips_file, "r", encoding="utf-8").read())
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(data, list):
        return []

    items = []
    for item in data:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "")).strip()
        name = str(item.get("name", "")).strip()
        if url:
            items.append({"name": name or url, "url": url})
    return items


def save_camera_ips(camera_ips_file: str, items: List[dict]):
    os.makedirs(os.path.dirname(camera_ips_file) or ".", exist_ok=True)
    with open(camera_ips_file, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def frame_to_base64(frame) -> str | None:
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def save_last_frame(last_frame_file: str, frame):
    os.makedirs(os.path.dirname(last_frame_file) or ".", exist_ok=True)
    cv2.imwrite(last_frame_file, frame)


def capture_camera_frame(url: str, capture_height: int, last_frame_file: str):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        cap.release()
        return {"error": "failed to open camera stream"}

    frame = read_non_black_frame(cap)
    cap.release()
    if frame is None:
        return {"error": "failed to read frame from camera"}

    frame = resize_frame_to_height(frame, capture_height)
    save_last_frame(last_frame_file, frame)

    image_b64 = frame_to_base64(frame)
    if image_b64 is None:
        return {"error": "failed to encode frame"}

    return {"status": "success", "image": image_b64}


def get_last_frame_b64(last_frame_file: str):
    if not os.path.exists(last_frame_file):
        return {"error": "last frame not found"}

    frame = cv2.imread(last_frame_file)
    if frame is None:
        return {"error": "failed to read last frame"}

    image_b64 = frame_to_base64(frame)
    if image_b64 is None:
        return {"error": "failed to encode frame"}

    return {"status": "success", "image": image_b64}
