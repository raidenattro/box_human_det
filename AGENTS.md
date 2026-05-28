# visual-dps — Agent 说明

## Docker 镜像 tag（必须）

- **不要**仅依赖 `:latest` 作为唯一交付 tag。
- 构建推理/UI 等镜像时，使用 `scripts/build-*.sh`；内部通过 `scripts/lib/docker-image-tag.sh` 生成：
  - 格式：`YYYYMMDD-HHMMSS-<git短哈希>`（同日多次构建可区分）
- 构建前导出：`VISUAL_DPS_IMAGE_TAG`（由 `scripts/lib/docker-build.sh` 自动设置）。
- 部署时在 `.env` 指定完整镜像名，例如：
  - `INFERENCE_LITE_GPU_ONNX_IMAGE=visual-dps-inference-lite-gpu-onnx:20260528-143052-5eae119`
- 若需额外保留 `latest`：`DOCKER_TAG_ALSO_LATEST=1 ./scripts/build-inference-lite-gpu-onnx-image.sh`

## 推理后端

- 已移除 OpenMMLab **mmpose** 运行时依赖。
- 当前仅 `rtmpose_onnx`；计划新增 `yolo_pose`（见 `docs/FUTURE-yolo-pose.md`，默认 **YOLO26s-pose**）。
- 页脚版本：`GET /api/version` → `UI · API · Event · Infer`；产品号来自根目录 `version.json`。

## 摄像头流配置

- `source_type`：`external` | `publisher` | `rtsp_pull` | `v4l2`
- **`rtsp_pull`**：`pull_url` = 上游拉流；`url` = 本机 MediaMTX 播放地址（勿混填）
