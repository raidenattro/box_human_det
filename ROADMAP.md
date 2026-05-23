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
