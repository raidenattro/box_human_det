# Visual-DPS 离线恢复包 — 快速说明

## 包内有什么

| 路径 | 说明 |
|------|------|
| `docker-images/bundle.tar` | 全部 Docker 镜像（`docker load` 用） |
| `app/` | `docker-compose.yml`、`.env`、`app_config.json`、`localdata/` |
| `install.sh` | 目标机一键：`docker load` + `compose up -d` |
| `PACKAGE_INFO.txt` | 打包时间、镜像列表 |
| `app/deploy/PACKAGE-MANIFEST.md` | 完整清单与检查表 |

## 目标机恢复（完全离线）

```bash
tar xzf visual-dps-offline-*.tar.gz   # 若收到的是 tar.gz
cd visual-dps-offline-*/

# 按需改 app/.env：REDIS_PASSWORD、MEDIAMTX_PUBLIC_HOST（本机 IP）
./install.sh
```

浏览器：`http://<MEDIAMTX_PUBLIC_HOST>:<UI_PORT>/`（默认 UI_PORT=8045）

## 权重「有文件」不够

`localdata/models/` 下 8 个权重须 **存在且体积 ≥ 约定下限**（见 `app/deploy/model-weights-spec.sh`）。  
`install.sh` 会校验；半截文件不算就绪。

## 为什么打包时又下载？

- **不会**对「源机已完整」的 `localdata/models` 重复联网下载，只会 **rsync 拷贝** 进包。
- 仅当源机缺文件/不完整、或包内 rsync 后仍不达标时，export 才 **只补缺失项**。
- 源机 `localdata/models` 若属 root（Docker 写入），请先：  
  `sudo chown -R "$USER:$USER" localdata/models`  
  再执行 `./scripts/download-model-weights.sh` 一次打全。

## 源机重新打完整包

```bash
./scripts/download-model-weights.sh          # 校验通过后再 export
./scripts/export-offline-package.sh --inference all
```
