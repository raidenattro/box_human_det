# 推理 / 事件 管道拆分（S1–S5 已完成）

## 目标拓扑

```
推理容器 (每路 1 个)          事件 Worker (默认 1 个，可按摄像头分片扩 N 个)
RTSP → 检测+17点姿态  ──pose──►  碰撞/报警/回调 Java
         │ Redis                    │
         └ pose:snapshot            └ event:snapshot
                    ↘              ↙
                     UI LiveHub → SSE frame
```

## Redis 契约

| 键 / 频道 | 说明 |
|-----------|------|
| `pose:live:{camera_id}` | Pub/Sub，原始姿态帧 |
| `pose:snapshot:{camera_id}` | 最新姿态（TTL 10s） |
| `event:live:{camera_id}` | Pub/Sub，碰撞/报警 overlay |
| `event:snapshot:{camera_id}` | 最新事件 overlay（TTL 10s） |

**PoseFrame** (`schema: 1`, `kind: "pose"`):

```json
{
  "schema": 1,
  "kind": "pose",
  "ts": 1710000000.0,
  "camera_id": "cam2",
  "frame_idx": 120,
  "infer_width": 640,
  "infer_height": 360,
  "persons": [{ "person_id": 0, "keypoints": [[x, y, score], ...] }]
}
```

**EventFrame** (`schema: 1`, `kind: "event"`):

```json
{
  "schema": 1,
  "kind": "event",
  "ts": 1710000000.0,
  "camera_id": "cam2",
  "frame_idx": 120,
  "collisions": ["shelf:box"],
  "alarm_collisions": ["shelf:box"]
}
```

**SSE `frame` 事件**（LiveHub 合并，兼容监控页）:

```json
{
  "schema": 2,
  "ts": ...,
  "infer_width": ...,
  "infer_height": ...,
  "frame_idx": ...,
  "skeletons": [...],
  "collisions": [...],
  "alarm_collisions": [...]
}
```

## 已移除（S5）

- `inference:live:*` / `inference:snapshot:*`
- `publish_live_frame()`（请用 `publish_pose_frame` / `publish_event_frame`）

## 部署

- `docker compose --profile ui up` 含 `visual-dps-event-worker`（默认 1 实例）
- 推理容器仅需 `REDIS_URL`；事件与回调在 event-worker 内

## 水平扩展（N 个 event-worker，共享队列 LB）

默认 `POSE_DELIVERY=stream`：推理 `XADD pose:stream`，事件 Worker 用 **Redis Stream Consumer Group**（`event-workers`）竞争消费——**谁空闲谁拿下一条 pose**（同一条不会重复处理）。

| 环境变量 | 默认 | 说明 |
|----------|------|------|
| `POSE_STREAM_KEY` | `pose:stream` | Stream 名 |
| `POSE_STREAM_GROUP` | `event-workers` | 消费组 |
| `EVENT_WORKER_CONSUMER_NAME` | 随机 | 组内消费者名，多实例须不同 |
| `POSE_DELIVERY` | `stream` | `pubsub` 为旧模式（可配合分片） |

多实例示例：

```bash
docker run -e EVENT_WORKER_CONSUMER_NAME=worker-a ... visual-dps-event-worker:latest
docker run -e EVENT_WORKER_CONSUMER_NAME=worker-b ... visual-dps-event-worker:latest
```

UI 实时仍走 `pose:live:{cam}` Pub/Sub + snapshot，与 Stream 队列并行。

旧分片（仅 `POSE_DELIVERY=pubsub` 时）：`EVENT_WORKER_SHARD_COUNT` / `INDEX`。

## RTSP 硬件解码（NVIDIA）

| 镜像 | 说明 |
|------|------|
| `visual-dps-inference-lite` | CPU ffmpeg / OpenCV 回退 |
| `visual-dps-inference-lite-gpu` | BtbN ffmpeg + `--gpus all`，启动时探测 `cuda`/`h264_cuvid` |

`RTSP_CAPTURE_BACKEND=auto`（默认）：有 NVIDIA 则用 ffmpeg 管道，否则 OpenCV。  
MediaPipe 仍在 CPU；加速的是 **解码**，不是 17 点推理。

## 构建加速

`docker-compose.yml` 注入 `APT_MIRROR`、`PIP_INDEX`（默认阿里云 apt + 清华 pip）。可复制 `.env.example` → `.env`。

## 性能说明

CPU MediaPipe 慢则姿态发布慢，事件 Worker 不会「补帧」；勿用重复发布旧数据等 hack。
