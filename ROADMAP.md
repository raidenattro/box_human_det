# 后续计划：统一配置 UI

## 目标

1. **UI 可改配置，尽量热生效** — 修改后无需重启服务（模型路径等设备级配置除外）
2. **配置入口集中** — 单页 Settings，不分散在 `/` 与 `/annotate`
3. **配置项尽量少** — 现场实施只暴露高频项，高级项用默认值

## 现场暴露配置（建议 5 项）

| 配置 | 字段 | 热生效 | 说明 |
|------|------|--------|------|
| 推理帧率 | `inference.frame_rate` | 是 | 默认 15 fps |
| 推理高度 | `inference.height` | 是 | 默认 480 px |
| 姿态间隔 | `inference.pose_frame_interval` | 是 | 默认 3 帧 |
| 视频流地址 | `source.stream_url` | 是* | 变更后重连流 |
| 预览开关 | `debug-info.enabled` | 是 | 关闭可显著降负载 |

其余项（`stream_buffer_size`、`preview_jpeg_quality`、`capture_height` 等）保持配置文件默认值，不在 UI 暴露。

## 实现要点

- 新增 `GET/PATCH /api/runtime_config`，读写内存配置并持久化到 `app_config.json`
- 推理循环每帧读取内存配置（frame_rate / height / pose_frame_interval）
- 标注页与监控页顶部共用 Settings 抽屉，不再跨页找配置
- 标注相关 `capture_height` 与上传转码高度跟随 `inference.height`，减少重复配置


## 讨论（已实现：见 [docs/PIPELINE_SPLIT.md](./docs/PIPELINE_SPLIT.md)）

1. **17 点姿态** — 每路独立推理容器，发布 `pose:live:{camera_id}`
2. **动作/事件** — 默认 1 个 `visual-dps-event-worker`；可按 `EVENT_WORKER_SHARD_COUNT/INDEX` 起 N 个（按摄像头分片，见 PIPELINE_SPLIT.md）
3. **UI** — `LiveHub` 合并 pose + event → SSE；推理容器不再做碰撞/回调

部署：`docker compose --profile ui up` 含 redis + ui + event-worker；按摄像头 `inference/start` 起推理容器。

## 新需求

1. 需要一个页面，把所有区域/货架全部显示出来，以Camera为单位/成组。我需要看到所有的事件被识别到的全局界面。可以是一个大表格/矩阵？

2. 需要澄清，开启检测后到底开启了什么？我docker ps没看到容器有增加；

3. 监控界面上的“开启检测”开关，旁边默认显示一横，表示全局没有配模型？但实际上全局配好了的，有些摄像头有问题，不全是。另外摄像头个性配置中选择了指定的算法，监控界面上也不会变，很奇怪！需要核实并解决。

4. 模型框架单核 FPS多核 FPS备注
RTMPose-tMMPose / ONNX~25–35~60–90最轻量，工业部署首选
RTMPose-sMMPose / ONNX~15–25~40–60精度/速度均衡
MoveNet LightningTFLite~25–40~50–80Google官方，移动端优化好