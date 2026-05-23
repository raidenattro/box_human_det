"""标注 JSON 读写服务。"""

import json
import os
from datetime import datetime

from core.state import STATE


def resolve_annotation_path(json_dir: str, default_json_file: str) -> str:
    if STATE.json_path:
        return STATE.json_path
    return default_json_file


def save_annotation(data: dict, default_json_file: str, json_dir: str):
    os.makedirs(json_dir, exist_ok=True)

    if STATE.json_path:
        json_path = STATE.json_path
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(json_dir, f"annotation_{timestamp}.json")
        STATE.json_path = json_path

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {"status": "success", "json_path": json_path}


def load_annotation(default_json_file: str, json_dir: str):
    json_path = resolve_annotation_path(json_dir, default_json_file)
    if not os.path.exists(json_path):
        return {"error": "annotation not found", "json_path": json_path}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {"error": f"invalid annotation json: {e}", "json_path": json_path}

    return {"status": "success", "data": data, "json_path": json_path}


def flatten_annotation_boxes(config_data: dict) -> list:
    if not isinstance(config_data, dict):
        return []

    raw_boxes = []
    shelves = config_data.get("shelves")
    if isinstance(shelves, list):
        for shelf in shelves:
            if not isinstance(shelf, dict):
                continue
            shelf_code = str(shelf.get("shelf_code", "") or "").strip()
            boxes = shelf.get("boxes", [])
            if not isinstance(boxes, list):
                continue
            for box in boxes:
                if not isinstance(box, dict):
                    continue
                item = dict(box)
                if shelf_code and not item.get("shelf_code"):
                    item["shelf_code"] = shelf_code
                raw_boxes.append(item)

    if raw_boxes:
        return raw_boxes

    boxes = config_data.get("boxes", [])
    return boxes if isinstance(boxes, list) else []
