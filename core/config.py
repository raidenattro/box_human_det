"""应用配置的严格加载与校验模块。

这个模块统一负责配置结构校验，确保运行时逻辑可以直接依赖
路径、模型地址和服务参数已经存在且合法。
"""

import json
import os

CONFIG_FILE = "app_config.json"

REQUIRED_CONFIG_KEYS = {
    "paths": [
        "templates_dir",
        "index_html",
        "base_localdata_dir",
        "upload_dir",
        "upload_480p_dir",
        "json_dir",
        "default_json_file",
        "counter_file",
    ],
    "models": [
        "device",
        "det_config",
        "det_checkpoint",
        "pose_config",
        "pose_checkpoint",
    ],
    "inference": ["det_interval", "pose_interval"],
    "server": ["host", "port"],
}


def _normalize_local_path(path_value: str) -> str:
    """规范化本地文件路径，但保留 URL 原样不动。"""
    if path_value.startswith("http://") or path_value.startswith("https://"):
        return path_value
    return os.path.normpath(path_value)


def _validate_config_or_raise(cfg: dict, config_file: str):
    """校验必填字段和取值范围，发现问题后直接退出。"""
    errors = []

    for section, keys in REQUIRED_CONFIG_KEYS.items():
        section_data = cfg.get(section)
        if not isinstance(section_data, dict):
            errors.append(f"缺少对象字段: {section}")
            continue

        for key in keys:
            if key not in section_data:
                errors.append(f"缺少字段: {section}.{key}")
                continue

            value = section_data[key]
            if isinstance(value, str) and not value.strip():
                errors.append(f"字段不能为空: {section}.{key}")

    try:
        det_interval = int(cfg["inference"]["det_interval"])
        if det_interval <= 0:
            errors.append("字段必须大于 0: inference.det_interval")
    except Exception:
        errors.append("字段必须是整数: inference.det_interval")

    try:
        pose_interval = int(cfg["inference"]["pose_interval"])
        if pose_interval <= 0:
            errors.append("字段必须大于 0: inference.pose_interval")
    except Exception:
        errors.append("字段必须是整数: inference.pose_interval")

    try:
        port = int(cfg["server"]["port"])
        if not (1 <= port <= 65535):
            errors.append("字段范围无效(1-65535): server.port")
    except Exception:
        errors.append("字段必须是整数: server.port")

    if errors:
        msg = [f"配置文件校验失败: {config_file}"]
        msg.extend([f"- {e}" for e in errors])
        raise SystemExit("\n".join(msg))


def load_app_config(config_file: str = CONFIG_FILE) -> dict:
    """加载 app_config.json 并强制校验其结构。

    这个过程是严格模式：文件不存在、格式错误或字段缺失都会直接
    中止启动，避免错误配置被带到运行时才暴露。
    """
    if not os.path.exists(config_file):
        raise SystemExit(f"配置文件不存在: {config_file}，请先创建后再启动。")

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    except json.JSONDecodeError as e:
        raise SystemExit(f"配置文件 JSON 格式错误: {config_file}，详情: {e}")
    except Exception as e:
        raise SystemExit(f"读取配置文件失败: {config_file}，详情: {e}")

    if not isinstance(loaded, dict):
        raise SystemExit(f"配置文件根节点必须是对象: {config_file}")

    _validate_config_or_raise(loaded, config_file)

    for k, v in loaded.get("paths", {}).items():
        if isinstance(v, str):
            loaded["paths"][k] = _normalize_local_path(v)

    for k, v in loaded.get("models", {}).items():
        if isinstance(v, str):
            loaded["models"][k] = _normalize_local_path(v)

    return loaded
