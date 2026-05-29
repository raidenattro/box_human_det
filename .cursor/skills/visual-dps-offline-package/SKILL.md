---
name: visual-dps-offline-package
description: >-
  Build and install Visual-DPS offline deployment packages: export-offline-package.sh,
  export-offline-complete.sh, download-model-weights.sh, install.sh, verify-package.sh.
  Use when the user mentions 离线包、离线部署、export-offline、docker save、weights、
  bundle.tar、全量包、install.sh、或目标机无网恢复 Visual-DPS。
---

# Visual-DPS 离线打包与安装

## 包布局（v2，默认）

| 路径 | 内容 |
|------|------|
| `docker-images/bundle.tar` | `docker save` 全部镜像（**不要 gzip**） |
| `app/` | `docker-compose.deploy.yml`、`.env`、`app_config.json`、`localdata`（**无 models**） |
| `weights/` | 8 个推理权重 + `SHA256SUMS` |
| `install.sh` / `verify-package.sh` | 目标机安装与校验 |

全量通用包 = **7 镜像**（redis、mediamtx、ui、event-worker、inference-lite、lite-gpu、lite-gpu-onnx）。

## 源机：打全量包（推荐）

```bash
cd <repo-root>

# 1. 权重（只需在源机做一次，或更新模型时重做）
./scripts/download-model-weights.sh

# 2. 确保本地已有全部镜像（按需 build）
./scripts/build-ui-image.sh
./scripts/build-inference-lite-image.sh
./scripts/build-inference-lite-gpu-image.sh
./scripts/build-inference-lite-gpu-onnx-image.sh

# 3. 打包容器（默认 dist/ 下目录，几分钟级）
./scripts/export-offline-complete.sh
```

输出：`dist/visual-dps-offline-complete-<YYYYMMDD-HHMMSS>/`

### 常用选项

| 选项 | 用途 |
|------|------|
| `--rebuild-ui` | 打包前重建 UI + event-worker 镜像 |
| `--archive tar` | 额外打未压缩 `.tar` 外壳（便于 U 盘） |
| `--archive tar.gz --compress pigz` | 外传用压缩包（**用 pigz，勿用默认 gzip 扫 14G**） |
| `--split 2G` | 对**外层归档**分卷 |
| `--inference lite\|gpu-onnx\|base\|all` | 控制打进 bundle 的推理镜像 |
| `--no-models` | 不打包 weights/ |
| `--allow-download-weights` | 源机不齐时：**先** `deploy/recover-model-weights.sh` 从 `dist/` 旧包拷贝，**再**联网补缺失项 |
| `FORCE_SAVE=1` | 忽略已有 `bundle.tar`，强制重做 `docker save` |
| `-o dist/my-pkg` | 指定输出目录 |

### 非全量示例

```bash
# 仅 Web + 事件（无推理镜像）
./scripts/export-offline-package.sh --inference base --no-models

# 仅 GPU-ONNX 推理 + 权重
./scripts/export-offline-package.sh --inference gpu-onnx
```

## 目标机：安装

```bash
cd <包根目录>
./verify-package.sh
# 必须修改 app/.env 中 REDIS_PASSWORD（不能用 change-me-before-install）
./install.sh --host <局域网IP> --stop-infer
```

| 选项 | 说明 |
|------|------|
| `--host IP` | 写入 `MEDIAMTX_PUBLIC_HOST` |
| `--weights-dir DIR` | 默认 `<包根>/weights`；旧包可用 `app/localdata/models` |
| `--stop-infer` | 删除旧 `visual-dps-infer-*` 容器 |

访问：`http://<MEDIAMTX_PUBLIC_HOST>:<UI_PORT>/`（默认 8045）

## 分发方式

| 场景 | 做法 |
|------|------|
| 内网 / U 盘 / rsync | 直接拷贝**目录**，或 `--archive tar` |
| 带宽有限 | `--archive tar.gz --compress pigz` 或 `zstd` |
| 仅更新权重 | 单独同步 `weights/`（~276M），不必重打 bundle |

```bash
# 153 测试机固定账号（勿用 xu@ / sugar@）
rsync -av --progress dist/visual-dps-offline-complete-*/ hqit@192.168.1.153:~/workspace/visual-dps-0529/
```

## 校验与排错

```bash
./verify-package.sh              # 包根执行
VERIFY_QUICK=1 ./verify-package.sh   # 跳过打印镜像 JSON

# 源机权重检查
source deploy/check-model-weights.sh
visual_dps_check_model_weights ./localdata
```

| 现象 | 处理 |
|------|------|
| export 报权重不全 | 源机 `./scripts/download-model-weights.sh` |
| 缺镜像 | 按 inference 模式 build 对应 Dockerfile 脚本 |
| complete 缺 lite 镜像 | 勿用 `.env` 的 `INFERENCE_LITE_IMAGE` 代替 lite；`all` 模式已固定收集三档 |
| install 报 REDIS_PASSWORD | 编辑 `app/.env` |
| 204 仅 docker-compose v1 | `install.sh` 已优先 `docker-compose.deploy.yml` |

## 相关文件

- `scripts/export-offline-package.sh` — 主入口
- `scripts/export-offline-complete.sh` — `--inference all` 快捷方式
- `scripts/download-model-weights.sh` — 源机权重
- `deploy/install.sh`、`deploy/verify-package.sh`
- `deploy/model-weights-spec.sh`、`deploy/check-model-weights.sh`
- `deploy/PACKAGE-MANIFEST.md`、`deploy/OFFLINE-QUICKSTART.md`

## 性能原则（agent 须遵守）

1. **默认 `--archive none`**：产出目录，不对 14G `bundle.tar` 做 `tar -czf`。
2. **权重永远在 `weights/`**，不参与外层 gzip。
3. 只有用户明确要求外传压缩包时，才建议 `--archive tar.gz --compress pigz`。
4. `--rebuild-ui` 仅在有 UI/后端变更时使用（约 +10 分钟）。
