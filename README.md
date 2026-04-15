# box_human_det 架构与联调说明

本文档用于梳理 `box_human_det` 当前代码结构、模块调度关系、与 `annotation-tool` 的协作方式，以及与 Java 后端的回调链路。

## 1. 项目定位

`box_human_det` 是一个 FastAPI 服务，负责：

- 接收视频源（本地上传或配置文件中的网络流）
- 读取货架标注 JSON
- 执行人体检测 + 姿态估计 + 手腕与货框碰撞判定
- 将报警事件异步回调给 Java 后端
- 通过 WebSocket 向前端推送可视化结果

## 2. 目录与职责

- `main.py`
  - 进程入口，启动 Uvicorn。
- `app_factory.py`
  - 组装应用、注册路由、创建服务实例。
- `core/config.py`
  - 加载并校验 `app_config.json`。
- `core/state.py`
  - 运行期共享状态 `STATE`。
- `services/video_service.py`
  - 本地视频上传与 480p 转码。
  - 从配置初始化网络流来源（不再通过接口切换流地址）。
- `services/annotation_service.py`
  - 保存标注 JSON。
- `services/inference_service.py`
  - 推理主循环、碰撞判定、报警触发、WebSocket 推送。
- `services/callback_reporter.py`
  - 报警事件异步队列与 Java 回调重试机制。
- `frontend/index.html`
  - 内置前端页面：上传、本地标注、启动推理、三屏可视化。
- `app_config.json` / `app_config.jsonc`
  - 运行配置。

## 3. 启动与初始化调用链

启动顺序：

1. `main.py` 调用 `create_app()`。
2. `app_factory.py` 加载配置 `load_app_config()`。
3. `app_factory.py` 调用 `initialize_source_from_config()`：
   - 若 `source.enabled=true` 且 `source.stream_url` 非空，则把网络流写入 `STATE.video_path`。
   - 同时确定推理要读的标注 JSON 路径 `STATE.json_path`。
4. 创建 `CollisionCallbackReporter` 与 `InferenceService`。
5. FastAPI `startup` 事件启动回调 worker。

## 4. 运行时共享状态（STATE）

`core/state.py` 中 `STATE` 是跨路由共享的当前会话状态，关键字段：

- `video_path`：当前视频源（本地视频路径或 RTSP/HTTP URL）
- `source_type`：`file` 或 `stream`
- `source_url`：网络流 URL
- `json_path`：当前会话标注 JSON 路径
- `upload_tag`：事件标识前缀（回调用）
- `is_inferencing`：推理开关

## 5. 视频源模式与调度关系

### 5.1 本地上传模式

调用 `POST /api/upload_video`：

- 视频先落盘到 `localdata/upload/`。
- 尝试转码到 480p（`localdata/upload/480p/`）。
- 为本次会话分配 `upload_tag` 与对应 `json_path`（`localdata/json/{upload_tag}_boxes.json`）。

### 5.2 配置网络流模式

通过 `app_config.json` 的 `source` 段初始化（不再使用运行时接口切换流）：

```json
"source": {
  "enabled": true,
  "stream_url": "rtsp://...",
  "upload_tag": "stream_config",
  "annotation_json": "D:/.../annotation-tool/annotation_json/annotation_xxx.json"
}
```

说明：

- `annotation_json` 支持绝对路径。
- 若填相对路径，会按 `paths.json_dir` 拼接。
- 若留空，回退 `paths.default_json_file`。

## 6. 标注 JSON 如何进入推理

推理启动时，`InferenceService.websocket_inference()` 会读取：

- `STATE.json_path`（优先）
- 或 `paths.default_json_file`（兜底）

读取后提取：

- `boxes`
- `annotation_size`（如果存在）
- `source_info`（如果存在）

### 6.1 坐标统一与换算

当前推理端已经做了统一处理：

1. 对网络流可按 `inference.stream_target_height` 统一推理分辨率（默认 480 高，等比缩放）。
2. 标注框优先使用 `video_polygon_norm` 映射到当前推理帧尺寸。
3. 若没有 `video_polygon_norm`，则尝试使用 `annotation_size` 与当前帧比例进行缩放。
4. 再兜底使用原始 `video_polygon`。

这样可以避免“标注分辨率”和“推理分辨率”不一致导致的碰撞偏移。

## 7. 推理主循环与报警机制

`services/inference_service.py` 核心流程：

1. 懒加载 MMDetection / MMPose 模型。
2. 读帧（本地文件或网络流）。
3. 按 `det_interval` 执行检测，按 `pose_interval` 执行姿态。
4. 提取手腕关键点，逐框做 `pointPolygonTest` 判定碰撞。
5. 连续命中 + 冷却帧门槛形成 `alarm_collisions`。
6. 报警事件入队 `callback_reporter`。
7. 通过 WebSocket 发送图像、骨架、碰撞、报警和统计信息。

## 8. 与 Java 后端交互

Java 交互由 `CollisionCallbackReporter` 负责：

- 推理线程只入队，不阻塞主循环。
- worker 异步 HTTP POST 到 `reporting.callback_url`。
- 支持重试、退避、队列上限、状态记录。
- 可通过 `GET /api/callback_report/{event_id}` 查询上报状态。

## 9. 与 annotation-tool 的交互边界

`box_human_det` 与 `annotation-tool` 没有直接 RPC 调用关系，交互通过“标注 JSON 文件”完成。

推荐联调方式：

1. 在 `annotation-tool` 完成摄像头抓帧标注并保存 JSON。
2. 在 `box_human_det/app_config.json` 中设置：
   - `source.enabled=true`
   - `source.stream_url=目标摄像头流`
   - `source.annotation_json=annotation-tool 生成 JSON 的绝对路径`
3. 启动 `box_human_det` 后直接推理。

注意：

- `annotation-tool` 生成的 JSON 通常包含 `source_info`、`annotation_size`、`video_polygon_norm`，更利于跨分辨率推理。
- `box_human_det` 自带前端保存的 JSON 可能只有 `video_polygon`，仍可运行，但坐标鲁棒性较弱。

## 10. 主要接口一览

- `GET /`：前端页面
- `POST /api/upload_video`：上传本地视频
- `GET /api/get_first_frame`：取当前视频首帧
- `POST /api/save_annotation`：保存标注 JSON
- `GET /api/get_current_annotation`：获取当前会话标注
- `POST /api/start_inference`：启动推理
- `GET /api/callback_report/{event_id}`：查询回调状态
- `WS /ws/inference`：实时推理推流

## 11. 常见问题

### Q1：为什么配置了网络流但还是没跑？

检查：

- `source.enabled` 是否为 `true`
- `source.stream_url` 是否非空且可连通
- `source.annotation_json` 是否存在
- 启动日志是否打印“已从配置文件初始化网络流”

### Q2：为什么提示找不到标注 JSON？

- 网络流模式下优先用 `source.annotation_json` 解析出的路径。
- 路径为空时回退 `default_json_file`。
- 相对路径是相对 `paths.json_dir`，不是仓库根目录。

### Q3：为什么碰撞位置偏了？

优先使用包含 `video_polygon_norm` 和 `annotation_size` 的标注 JSON。

---

如果后续要做严格生产化，建议继续补两项：

1. 在推理启动前做“标注来源 camera_url 与当前 stream_url 一致性硬校验”。
2. 在配置加载阶段对 `source.annotation_json` 做存在性校验并给出更早期错误提示。
