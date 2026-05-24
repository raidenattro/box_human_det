"""将 Vite 构建产物挂载为 SPA 静态站点。"""

import os

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def mount_spa(app: FastAPI, dist_dir: str) -> None:
    dist_dir = os.path.abspath(dist_dir)
    index_path = os.path.join(dist_dir, "index.html")
    assets_dir = os.path.join(dist_dir, "assets")

    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"SPA index not found: {index_path}")

    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="spa-assets")

    for filename in ("favicon.ico", "favicon.svg", "apple-touch-icon.png"):
        filepath = os.path.join(dist_dir, filename)
        if not os.path.isfile(filepath):
            continue

        def _make_handler(file_path: str):
            async def _handler():
                return FileResponse(file_path)

            return _handler

        app.get(f"/{filename}", name=f"spa-{filename}")(_make_handler(filepath))

    @app.get("/")
    async def spa_root():
        return FileResponse(index_path)

    @app.get("/login")
    async def spa_login():
        return FileResponse(index_path)

    @app.get("/monitor")
    async def spa_monitor():
        return FileResponse(index_path)

    @app.get("/annotate")
    async def spa_annotate():
        return FileResponse(index_path)

    @app.get("/settings")
    async def spa_settings():
        return FileResponse(index_path)
