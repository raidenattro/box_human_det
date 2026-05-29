# Visual-DPS 离线包（v2 布局）

## 目录结构

```text
visual-dps-offline-complete-<时间戳>/
├── docker-images/bundle.tar   # docker save（勿再 gzip）
├── app/                       # compose、.env、配置（无 models）
├── weights/                   # 8 个权重 + SHA256SUMS
├── install.sh
├── verify-package.sh
└── PACKAGE_INFO.txt
```

## 源机打全量包

```bash
./scripts/download-model-weights.sh
./scripts/export-offline-complete.sh
# 产出: dist/visual-dps-offline-complete-<ts>/

# 需重建 UI 镜像时
./scripts/export-offline-complete.sh --rebuild-ui

# 外传可选（pigz 多核，比 tar -czf 快很多）
./scripts/export-offline-complete.sh --archive tar.gz --compress pigz

# 仅未压缩 tar 外壳（仍不 gzip bundle）
./scripts/export-offline-complete.sh --archive tar
```

## 目标机安装

```bash
# 若用了 --archive tar.gz
tar xzf visual-dps-offline-complete-*.tar.gz

cd visual-dps-offline-complete-*/
./verify-package.sh
# 编辑 app/.env 设置 REDIS_PASSWORD
./install.sh --host <本机IP> --stop-infer
```

权重默认从 `weights/` 安装到 `app/localdata/models/`。旧版单 tar.gz 包若含 `app/localdata/models` 仍兼容。

## 体积说明

| 部分 | 约 |
|------|-----|
| bundle.tar（7 镜像） | ~14G |
| weights/ | ~276M |
| app/ | 数 MB |

慢的根源是对 14G 做 `tar -czf`；v2 **默认不压整包**，只 rsync/拷贝目录。

详细命令见 `.cursor/skills/visual-dps-offline-package/SKILL.md`。
