# box_human_det Docker 部署

## 1. 前置条件

- Docker Desktop (Windows, WSL2 backend)
- 已启用 GPU 支持（如需 CUDA 推理）
- 保证宿主机目录存在：
  - `../mmpose`
  - `../annotation-tool/annotation_json`

## 2. 配置文件

容器会把 `app_config.docker.json` 挂载为运行配置文件：`/app/app_config.json`。

你至少需要改这两项：

- `source.stream_url`
- `source.annotation_json`

## 3. 构建与启动

在 `box_human_det` 目录执行：

```bash
docker compose build
docker compose up -d
```

查看日志：

```bash
docker compose logs -f box-human-det
```

## 4. 访问服务

- 前端页面: `http://localhost:8045`
- WebSocket: `ws://localhost:8045/ws/inference`

## 5. 常见问题

### 5.1 容器中找不到配置文件

确认 `docker-compose.yml` 里有挂载：

- `./app_config.docker.json:/app/app_config.json:ro`

### 5.2 模型配置路径不存在

确认 `../mmpose` 目录已挂载到容器的 `/workspace/mmpose`。

### 5.3 RTSP 连接不上

Windows Docker Desktop 下，访问宿主机可用 `host.docker.internal`；
如果 RTSP 服务在别的机器，改成实际 IP 即可。

### 5.4 无法使用 GPU

- Docker Desktop 需启用 WSL2
- NVIDIA 驱动和 CUDA runtime 需兼容
- `docker compose` 输出里应看到 `gpus: all` 生效
