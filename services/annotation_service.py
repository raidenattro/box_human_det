"""保存前端生成的标注数据。

前端负责计算货架外框和网格坐标，这个模块只负责把结果写入当前
会话对应的 JSON 文件。
"""

import json
import os

from core.state import STATE


def save_annotation(data: dict, default_json_file: str, json_dir: str):
    """将当前会话的标注 JSON 持久化到磁盘。"""
    os.makedirs(json_dir, exist_ok=True)
    json_path = STATE.json_path or default_json_file
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    return {"status": "success", "json_path": json_path}
