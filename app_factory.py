"""FastAPI 应用工厂与路由绑定模块。

这个模块只负责把 HTTP / WebSocket 路由挂到应用上，真正的业务逻辑
都下沉到 services 层，避免入口文件继续膨胀。
"""

from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.responses import HTMLResponse

from core.config import load_app_config
from core.state import STATE
from services.annotation_service import save_annotation
from services.inference_service import InferenceService
from services.video_service import get_first_frame_b64, handle_video_upload


def create_app():
    """创建 FastAPI 应用并绑定所有路由。

    返回值：
        tuple: (app, app_config)，启动入口可以直接复用已加载的配置。
    """
    app_config = load_app_config()
    inference_service = InferenceService(app_config, STATE)
    app = FastAPI()

    @app.get("/", response_class=HTMLResponse)
    async def read_root():
        """返回前端入口页面。"""
        with open(app_config["paths"]["index_html"], "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)

    @app.post("/upload_video")
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

    @app.get("/get_first_frame")
    async def get_first_frame():
        """返回当前视频的第一帧，编码为 base64 JPEG。"""
        return get_first_frame_b64(STATE.video_path)

    @app.post("/save_annotation")
    async def save_annotation_api(data: dict):
        """保存前端生成的标注 JSON。"""
        return save_annotation(data, app_config["paths"]["default_json_file"], app_config["paths"]["json_dir"])

    @app.post("/start_inference")
    async def start_inference():
        """按需加载模型，并将会话切换到推理状态。"""
        return await inference_service.start_inference()

    @app.websocket("/ws/inference")
    async def websocket_inference(websocket: WebSocket):
        """通过 WebSocket 推送推理帧和运行数据。"""
        await inference_service.websocket_inference(websocket)

    return app, app_config
