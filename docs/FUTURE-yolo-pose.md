# 后续：YOLO Pose 后端（实验分支规划）

移除 OpenMMLab **mmpose** 后，计划新增 `yolo_pose` 推理后端（与 `mediapipe`、`rtmpose_onnx` 并列）。

## 推荐模型（已确认）

| 场景 | 模型 | 说明 |
|------|------|------|
| **默认** | **YOLO26s-pose** | 精度/速度平衡，适合多路 RTSP + GPU |
| 更高精度 | YOLO26m-pose | 路数较少或单路质检 |
| 极致路数 | YOLO26n-pose | 延迟优先 |

参考：Ultralytics YOLO26-pose（COCO keypoints，端到端，支持 ONNX/TensorRT 导出）。

## 实现要点（延后）

- 依赖：`ultralytics`（不引入 mmpose/mmengine）
- 镜像：在 `lite-gpu-onnx` 同类增量 Dockerfile 中安装，或独立 `Dockerfile.inference-lite-gpu-yolo`
- 与现有 `inference_container_service` 选图逻辑对齐 `INFERENCE_USE_GPU`
