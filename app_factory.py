"""FastAPI 应用工厂与路由绑定模块。

这个模块只负责把 HTTP / WebSocket 路由挂到应用上，真正的业务逻辑
都下沉到 services 层，避免入口文件继续膨胀。
"""

import json
import os

from fastapi import APIRouter, FastAPI, File, UploadFile, WebSocket
from fastapi.responses import FileResponse

from core.config import load_app_config
from core.state import STATE
from services.annotation_service import save_annotation
from services.callback_reporter import CollisionCallbackReporter
from services.inference_service import InferenceService
from services.video_service import get_first_frame_b64, handle_video_upload, initialize_source_from_config


def create_app():
    """创建 FastAPI 应用并绑定所有路由。

    返回值：
        tuple: (app, app_config)，启动入口可以直接复用已加载的配置。
    """
    app_config = load_app_config()
    initialize_source_from_config(
        app_config=app_config,
        json_dir=app_config["paths"]["json_dir"],
        default_json_file=app_config["paths"]["default_json_file"],
    )
    callback_reporter = CollisionCallbackReporter(app_config.get("reporting", {}))
    inference_service = InferenceService(app_config, STATE, callback_reporter=callback_reporter)
    app = FastAPI()
    api_router = APIRouter(prefix="/api")

    @app.on_event("startup")
    async def startup_event():
        await callback_reporter.start()

    @app.on_event("shutdown")
    async def shutdown_event():
        await callback_reporter.stop()

    @app.get("/")
    async def read_root():
        """返回前端入口页面。"""
        return FileResponse(app_config["paths"]["index_html"])

    @api_router.post("/upload_video")
    async def upload_video(file: UploadFile = File(...)):
        """接收新视频上传，并准备派生文件。"""
        return await handle_video_upload(
            file=file,
            upload_dir=app_config["paths"]["upload_dir"],
            upload_480p_dir=app_config["paths"]["upload_480p_dir"],
            json_dir=app_config["paths"]["json_dir"],
            counter_file=app_config["paths"]["counter_file"],
            base_localdata_dir=app_config["paths"]["base_localdata_dir"],
        )

    @api_router.get("/get_first_frame")
    async def get_first_frame():
        """返回当前视频的第一帧，编码为 base64 JPEG。"""
        return get_first_frame_b64(STATE.video_path)

    @api_router.post("/save_annotation")
    async def save_annotation_api(data: dict):
        """保存前端生成的标注 JSON。"""
        return save_annotation(data, app_config["paths"]["default_json_file"], app_config["paths"]["json_dir"])

    @api_router.post("/start_inference")
    async def start_inference():
        """按需加载模型，并将会话切换到推理状态。"""
        return await inference_service.start_inference()

    @api_router.get("/get_current_annotation")
    async def get_current_annotation():
        """返回当前会话将使用的标注 JSON，供前端启动前刷新显示。"""
        json_file_path = STATE.json_path or app_config["paths"]["default_json_file"]
        if not os.path.exists(json_file_path):
            return {"status": "error", "message": "annotation json not found", "json_path": json_file_path}

        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                config_data = json.load(f)
        except Exception as e:
            return {"status": "error", "message": f"failed to read annotation json: {e}", "json_path": json_file_path}

        boxes = config_data.get("boxes", []) if isinstance(config_data, dict) else []
        source_info = config_data.get("source_info", {}) if isinstance(config_data, dict) else {}
        if not isinstance(source_info, dict):
            source_info = {}
        return {
            "status": "success",
            "json_path": json_file_path,
            "boxes": boxes,
            "grid_shape": config_data.get("grid_shape", []) if isinstance(config_data, dict) else [],
            "source_info": source_info,
        }

    @api_router.get("/callback_report/{event_id}")
    async def get_callback_report(event_id: str):
        """查询算法拣货完成回调上报状态及 Java 响应。"""
        rec = callback_reporter.get_record(event_id)
        if rec is None:
            return {"status": "not_found", "event_id": event_id}
        return {"status": "ok", "record": rec}

    @app.websocket("/ws/inference")
    async def websocket_inference(websocket: WebSocket):
        """通过 WebSocket 推送推理帧和运行数据。"""
        await inference_service.websocket_inference(websocket)

    app.include_router(api_router)

    return app, app_config
