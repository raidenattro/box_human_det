# visual-dps

视觉拣货监护服务：货架区域标注、人体姿态推理、手腕/货框碰撞检测、拣货完成回调。

合并自 `box_human_det`（推理）与 `annotation_tool`（标注），统一为单个 FastAPI 服务。

## 页面

| 路径 | 说明 |
|------|------|
| `/` | 推理监控（上传视频、启动推理、可视化） |
| `/annotate` | 标注配置（多货架、摄像头 IP、抓帧标注） |

## 配置说明（`app_config.json`）

### video — 视频采集/转码

| 字段 | 含义 |
|------|------|
| `transcode_height` | 上传视频转码目标高度（像素，等比缩放） |
| `capture_height` | 抓帧/首帧用于标注的目标高度（像素） |

### inference — 推理管线

| 字段 | 含义 |
|------|------|
| `frame_rate` | 推理处理帧率（fps），限制每秒处理帧数 |
| `height` | 推理输入高度（像素，等比缩放，不放大） |
| `pose_frame_interval` | 姿态估计帧间隔（每 N 帧运行一次关键点模型） |
| `stream_buffer_size` | 网络流解码缓冲帧数（1=仅保留最新帧，降低 RTSP 延迟） |
| `preview_max_width` | WebSocket 预览推送最大宽度（像素） |
| `preview_jpeg_quality` | 预览 JPEG 质量（1–100） |

推理行为：

- 每个 processed 帧都执行人体检测
- 姿态估计按 `pose_frame_interval` 跳帧，中间帧复用上次骨架结果
- 可视化关闭时不编码 JPEG，仅后台推理与回调

### source — 启动时视频源

```json
"source": {
  "enabled": true,
  "stream_url": "rtsp://...",
  "upload_tag": "stream_config",
  "annotation_json": "localdata/json/annotation_xxx.json"
}
```

## 启动

```bash
pip install openmim
mim install "mmengine>=0.10.0" "mmcv>=2.0.0,<2.3.0" "mmdet>=3.0.0,<3.4.0"
pip install mmpose fastapi "uvicorn[standard]" python-multipart opencv-python-headless numpy psutil

python main.py
```

Docker：

```bash
docker compose up -d --build
```

## 主要 API

- `POST /api/upload_video` — 上传视频
- `GET /api/get_first_frame` — 获取首帧
- `GET/POST/DELETE /api/camera_ips` — 摄像头地址管理
- `POST /api/get_camera_frame` — 摄像头抓帧
- `POST /api/save_annotation` — 保存标注 JSON
- `GET /api/annotation` — 读取标注 JSON
- `POST /api/start_inference` — 启动推理
- `WS /ws/inference` — 实时推理推流

## 标注 JSON 格式

支持两种格式：

1. 旧格式：顶层 `boxes[]`
2. 新格式（多货架）：`shelves[].boxes[]`，含 `annotation_size`、`source_info`、`video_polygon_norm`

## 后续计划

统一配置 UI（热生效、单页集中、最少配置项）见 [ROADMAP.md](./ROADMAP.md)。
