# Visual-DPS 离线恢复包 — 快速说明

## 包内有什么

| 路径 | 说明 |
|------|------|
| `docker-images/bundle.tar` | 全部 Docker 镜像（或 `bundle.tar.part-*` 分卷需先合并） |
| `app/docker-compose.deploy.yml` | **离线安装用这个**（仅常驻服务） |
| `app/.env` | 部署配置（install 可 `--host` 改写） |
| `install.sh` | `docker load` + compose up |
| `verify-package.sh` | dry 校验（不启动服务） |
| `PACKAGE_INFO.txt` | 镜像列表、git 提交 |

## 目标机恢复

```bash
# 若分卷
cat visual-dps-offline-complete-*.tar.gz.part-* > visual-dps-offline-complete.tar.gz

tar xzf visual-dps-offline-complete-*.tar.gz
cd visual-dps-offline-complete-*/

./verify-package.sh          # 可选
./install.sh --host 192.168.0.204 --stop-infer
```

浏览器：`http://<MEDIAMTX_PUBLIC_HOST>:<UI_PORT>/`（默认 8045）

## GPU 机 `.env` 要点

```bash
MEDIAMTX_PUBLIC_HOST=<本机局域网 IP>
INFERENCE_USE_GPU=1
INFERENCE_LITE_GPU_ONNX_IMAGE=visual-dps-inference-lite-gpu-onnx:latest
```

## 权重

8 个文件须**存在且体积达标**（见 `app/deploy/model-weights-spec.sh`）。`install.sh` 会校验。

## 源机打完整包

```bash
./scripts/download-model-weights.sh
./scripts/export-offline-complete.sh --split 2G   # 可选分卷
```

## 说明

- GPU 镜像内 FFmpeg：构建时优先 BtbN CUDA 版，失败则回退 apt `ffmpeg`（推理仍可用）。
- 推理容器由 UI 按路 `docker run` 拉起，不在 compose 常驻栈内。
