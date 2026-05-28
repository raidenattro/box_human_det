"""部署组件版本（Docker 镜像 tag 等）。"""

from __future__ import annotations

import os

from core.product_version import format_product_version, short_image_ref


def _docker_client():
    try:
        import docker

        return docker.from_env()
    except Exception:
        return None


def _running_container_image(container_name: str) -> str:
    client = _docker_client()
    if not client:
        return ""
    try:
        c = client.containers.get(container_name)
        tags = list(c.image.tags or [])
        if tags:
            return tags[0]
        return str(c.image.short_id or "")
    except Exception:
        return ""


def _image_exists(client, image: str) -> bool:
    import docker

    if not str(image or "").strip():
        return False
    try:
        client.images.get(image)
        return True
    except docker.errors.ImageNotFound:
        return False


def _first_local_image(client, repo: str) -> str:
    repo = str(repo or "").strip()
    if not repo:
        return ""
    try:
        found = client.images.list(name=repo)
    except Exception:
        return ""
    tags: list[str] = []
    for img in found:
        tags.extend(img.tags or [])
    candidates = [t for t in tags if t.startswith(f"{repo}:")]
    if not candidates:
        return ""
    dated = [t for t in candidates if ":latest" not in t]
    return sorted(dated or candidates)[-1]


def _resolve_inference_image_ref() -> str:
    client = _docker_client()
    onnx_env = os.environ.get("INFERENCE_LITE_GPU_ONNX_IMAGE", "").strip()
    gpu_env = os.environ.get("INFERENCE_LITE_GPU_IMAGE", "").strip()
    lite_env = os.environ.get("INFERENCE_LITE_IMAGE", "").strip()
    if not client:
        return onnx_env or gpu_env or lite_env
    use_gpu = os.environ.get("INFERENCE_USE_GPU", "0") == "1"
    if use_gpu:
        for candidate in (
            onnx_env,
            _first_local_image(client, "visual-dps-inference-lite-gpu-onnx"),
            gpu_env,
            _first_local_image(client, "visual-dps-inference-lite-gpu"),
        ):
            if candidate and _image_exists(client, candidate):
                return candidate
    for candidate in (
        lite_env,
        _first_local_image(client, "visual-dps-inference-lite"),
    ):
        if candidate and _image_exists(client, candidate):
            return candidate
    return onnx_env or gpu_env or lite_env


def get_deployment_versions() -> dict:
    product = format_product_version()
    ui_image = (
        os.environ.get("VISUAL_DPS_UI_IMAGE", "").strip()
        or _running_container_image("visual-dps-ui")
    )
    event_image = (
        os.environ.get("VISUAL_DPS_EVENT_WORKER_IMAGE", "").strip()
        or _running_container_image("visual-dps-event-worker")
    )
    infer_image = _resolve_inference_image_ref()

    components = {
        "product": product,
        "ui_api": product,
        "ui_image": ui_image or "—",
        "event_worker": event_image or "—",
        "inference": infer_image or "—",
    }

    display_parts = [
        f"UI/API {product}",
        f"Event {short_image_ref(event_image)}",
        f"Infer {short_image_ref(infer_image)}",
    ]
    return {
        "status": "success",
        "product": product,
        "components": components,
        "display": " · ".join(display_parts),
    }
