# visual-dps — Agent 说明

## Docker 镜像 tag（必须）

- **不要**仅依赖 `:latest` 作为唯一交付 tag。
- 构建推理/UI 等镜像时，使用 `scripts/build-*.sh`；内部通过 `scripts/lib/docker-image-tag.sh` 生成：
  - 格式：`YYYYMMDD-HHMMSS-<git短哈希>`（同日多次构建可区分）
- 构建前导出：`VISUAL_DPS_IMAGE_TAG`（由 `scripts/lib/docker-build.sh` 自动设置）。
- 部署时在 `.env` 指定完整镜像名，例如：
  - `INFERENCE_LITE_GPU_ONNX_IMAGE=visual-dps-inference-lite-gpu-onnx:20260528-143052-5eae119`
  - `VISUAL_DPS_IMAGE_TAG=20260528-143052-5eae119`（`visual-dps-ui` / `visual-dps-event-worker` 共用，由 `build-ui-image.sh` 写入 `.env`）
- 若需额外保留 `latest`：`DOCKER_TAG_ALSO_LATEST=1 ./scripts/build-inference-lite-gpu-onnx-image.sh`（UI/Event 构建脚本同样支持）

## 推理后端

- 已移除 OpenMMLab **mmpose** 运行时依赖。
- `models.backend` 预设：`rtmpose_t|s|m`、`yolo26n|s|m|l_pose`（见 `services/inference_backends/model_registry.py`）。
- 页脚版本：`GET /api/version` → `UI · API · Event · Infer`；产品号来自根目录 `version.json`。

## 摄像头流配置

- `source_type`：`publisher`（外部推流）| `rtsp_pull`（拉取外部流）| `external`（直连 RTSP）
- **`rtsp_pull`**：`pull_url` = 上游拉流；`url` = 本机 MediaMTX 播放地址（勿混填）

## 改动后构建部署（必须）

Agent 改完 **web/**、**services/**、**core/** 后，**同一轮**执行构建并同步到运行中的 `visual-dps-ui`，再汇报；勿只改仓库不部署。

```bash
(cd web && npm run build)
docker cp web/dist/. visual-dps-ui:/app/web/dist/
# Python 变更：docker cp 对应文件到容器 /app/...
```

改 Dockerfile/依赖时用 `./scripts/build-ui-image.sh`（或 `--up`）。细则见 `.cursor/rules/build-deploy-after-change.mdc`。

## 153 异地机（固定）

- **SSH**：`hqit@192.168.1.153`（勿用 `xu@` / `sugar@`）
- **常用目录**：`~/workspace/visual-dps` 或测试包 `~/workspace/visual-dps-0529`
- **rsync 示例**：`rsync -av --progress dist/visual-dps-offline-complete-*/ hqit@192.168.1.153:~/workspace/visual-dps-0529/`
