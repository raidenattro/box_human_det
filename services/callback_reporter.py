"""碰撞回调上报服务。

设计目标：
1. 推理主循环只负责入队，不等待网络返回。
2. 后台 worker 异步发送到 Java 回调接口。
3. 保存每条上报状态，供 API 查询 Java 响应。
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib import error, request


@dataclass
class ReportRecord:
    event_id: str
    status: str
    payload: dict[str, Any]
    created_at: str
    updated_at: str
    http_status: int | None = None
    response_body: Any = None
    error: str | None = None
    retry_count: int = 0


class CollisionCallbackReporter:
    """将碰撞事件异步回调到 Java 后端。"""

    @staticmethod
    def _resolve_callback_url(cfg: dict) -> str:
        """优先按 callback_ip 配置拼装 URL，未配置时回退 callback_url。"""
        callback_ip = str(cfg.get("callback_ip", "")).strip()
        callback_url = str(cfg.get("callback_url", "")).strip()

        if not callback_ip:
            return callback_url

        callback_scheme = str(cfg.get("callback_scheme", "http")).strip() or "http"
        callback_path = str(cfg.get("callback_path", "/api/pick/finish")).strip() or "/api/pick/finish"
        callback_port_raw = str(cfg.get("callback_port", "")).strip()

        if not callback_path.startswith("/"):
            callback_path = f"/{callback_path}"

        host_part = callback_ip
        if callback_port_raw:
            try:
                host_part = f"{callback_ip}:{int(callback_port_raw)}"
            except Exception:
                host_part = f"{callback_ip}:{callback_port_raw}"

        return f"{callback_scheme}://{host_part}{callback_path}"

    def __init__(self, reporting_cfg: dict | None = None):
        cfg = reporting_cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.callback_url = self._resolve_callback_url(cfg)
        self.task_id = str(cfg.get("task_id", "")).strip()
        self.shelf_code = str(cfg.get("shelf_code", "")).strip()
        self.point_code = str(cfg.get("point_code", "")).strip()
        self.request_timeout_ms = int(cfg.get("request_timeout_ms", 800))
        self.queue_size = int(cfg.get("queue_size", 1000))
        self.cooldown_ms = int(cfg.get("cooldown_ms", 500))
        self.retry_times = int(cfg.get("retry_times", 2))
        self.retry_backoff_ms = int(cfg.get("retry_backoff_ms", 100))
        self.max_records = int(cfg.get("max_records", 5000))
        self.payload_template = cfg.get("payload_template")
        self.static_fields = cfg.get("static_fields", {})

        self._queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue(maxsize=self.queue_size)
        self._worker_task: asyncio.Task | None = None
        self._running = False

        self._records: dict[str, ReportRecord] = {}
        self._last_sent_at_ms_by_key: dict[str, int] = {}

        if self.enabled and not self.callback_url:
            print("⚠️ 回调上报已启用但 callback_url 为空，自动关闭上报")
            self.enabled = False

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    async def start(self):
        if not self.enabled or self._running:
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
        print("✅ 碰撞回调上报 worker 已启动")

    async def stop(self):
        if not self._running:
            return
        self._running = False
        if self._worker_task is not None:
            await self._queue.join()
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        print("✅ 碰撞回调上报 worker 已停止")

    def get_record(self, event_id: str) -> dict[str, Any] | None:
        rec = self._records.get(event_id)
        if rec is None:
            return None
        return {
            "event_id": rec.event_id,
            "status": rec.status,
            "payload": rec.payload,
            "created_at": rec.created_at,
            "updated_at": rec.updated_at,
            "http_status": rec.http_status,
            "response_body": rec.response_body,
            "error": rec.error,
            "retry_count": rec.retry_count,
        }

    def enqueue_pick_finished(
        self,
        box_id: Any,
        frame_idx: int,
        video_time_sec: float,
        upload_tag: str,
        shelf_code: str | None = None,
    ) -> str | None:
        """入队一个算法拣货完成回调事件；如果被节流或队列满则返回 None。"""
        if not self.enabled:
            return None

        now_ms = int(time.time() * 1000)
        dedupe_key = f"{upload_tag}:{box_id}"
        last_ms = self._last_sent_at_ms_by_key.get(dedupe_key)
        if last_ms is not None and now_ms - last_ms < self.cooldown_ms:
            return None

        event_id = uuid.uuid4().hex
        box_id_text = str(box_id).strip()
        if not box_id_text:
            return None
        context = {
            "event_id": event_id,
            "event_type": "ALGO_PICK_FINISHED",
            "upload_tag": upload_tag,
            "box_id": box_id_text,
            "frame_idx": int(frame_idx),
            "video_time_sec": round(float(video_time_sec), 3),
            "detected_at_ms": now_ms,
            "task_id": self.task_id or upload_tag,
            "shelf_code": (str(shelf_code).strip() if shelf_code is not None else "") or self.shelf_code or "",
            # pointCode 按当前命中的货框 box_id 动态传输，避免写死配置值。
            "point_code": box_id_text,
            "status": 1,
            "finish_time": now_ms,
        }
        payload = self._build_payload(context)

        record = ReportRecord(
            event_id=event_id,
            status="QUEUED",
            payload=payload,
            created_at=self._now_iso(),
            updated_at=self._now_iso(),
        )
        self._records[event_id] = record
        self._last_sent_at_ms_by_key[dedupe_key] = now_ms
        self._trim_records_if_needed()

        try:
            self._queue.put_nowait((event_id, payload))
            return event_id
        except asyncio.QueueFull:
            record.status = "DROPPED"
            record.error = "queue_full"
            record.updated_at = self._now_iso()
            return None

    def _trim_records_if_needed(self):
        overflow = len(self._records) - self.max_records
        if overflow <= 0:
            return
        for k in list(self._records.keys())[:overflow]:
            self._records.pop(k, None)

    def _build_payload(self, context: dict[str, Any]) -> dict[str, Any]:
        """按模板生成请求体；未配置模板时使用默认字段。"""
        if not isinstance(self.payload_template, dict) or not self.payload_template:
            payload = {
                "taskId": context["task_id"],
                "shelfCode": context["shelf_code"],
                "pointCode": context["point_code"],
                "status": int(context["status"]),
                "finishTime": int(context["finish_time"]),
            }
        else:
            payload = self._render_template_obj(self.payload_template, context)

        if isinstance(self.static_fields, dict):
            payload.update(self.static_fields)
        return payload

    def _render_template_obj(self, value: Any, context: dict[str, Any]) -> Any:
        if isinstance(value, dict):
            return {k: self._render_template_obj(v, context) for k, v in value.items()}
        if isinstance(value, list):
            return [self._render_template_obj(v, context) for v in value]
        if isinstance(value, str):
            # 纯占位符时保留上下文原始类型（如 int/bool），避免被强转成字符串。
            if value.startswith("{") and value.endswith("}") and value.count("{") == 1 and value.count("}") == 1:
                key = value[1:-1]
                if key in context:
                    return context[key]

            rendered = value
            for k, v in context.items():
                rendered = rendered.replace("{" + k + "}", str(v))
            return rendered
        return value

    async def _worker_loop(self):
        while self._running:
            event_id, payload = await self._queue.get()
            try:
                await self._send_with_retry(event_id, payload)
            finally:
                self._queue.task_done()

    async def _send_with_retry(self, event_id: str, payload: dict[str, Any]):
        rec = self._records.get(event_id)
        if rec is None:
            return

        for attempt in range(self.retry_times + 1):
            rec.status = "SENDING" if attempt == 0 else "RETRYING"
            rec.retry_count = attempt
            rec.updated_at = self._now_iso()

            print(
                f"[CALLBACK][SEND] event_id={event_id} attempt={attempt + 1}/{self.retry_times + 1} "
                f"url={self.callback_url} payload={json.dumps(payload, ensure_ascii=False)}"
            )

            ok, http_status, response_body, err = await asyncio.to_thread(self._post_json, payload)

            if ok:
                rec.status = "ACK"
                rec.http_status = http_status
                rec.response_body = response_body
                rec.error = None
                rec.updated_at = self._now_iso()
                print(
                    f"[CALLBACK][ACK] event_id={event_id} status={http_status} "
                    f"response={json.dumps(response_body, ensure_ascii=False)}"
                )
                return

            rec.http_status = http_status
            rec.response_body = response_body
            rec.error = err
            rec.updated_at = self._now_iso()

            is_client_error = http_status is not None and 400 <= http_status < 500
            if is_client_error:
                rec.status = "REJECT"
                print(
                    f"[CALLBACK][REJECT] event_id={event_id} status={http_status} "
                    f"error={err} response={json.dumps(response_body, ensure_ascii=False)}"
                )
                return

            if attempt < self.retry_times:
                print(
                    f"[CALLBACK][RETRY] event_id={event_id} status={http_status} "
                    f"error={err} next_in_ms={self.retry_backoff_ms * (2 ** attempt)}"
                )
                await asyncio.sleep((self.retry_backoff_ms * (2 ** attempt)) / 1000.0)

        rec.status = "FAILED"
        rec.updated_at = self._now_iso()
        print(
            f"[CALLBACK][FAILED] event_id={event_id} status={rec.http_status} "
            f"error={rec.error} response={json.dumps(rec.response_body, ensure_ascii=False)}"
        )

    def _post_json(self, payload: dict[str, Any]) -> tuple[bool, int | None, Any, str | None]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.callback_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        timeout_sec = max(self.request_timeout_ms / 1000.0, 0.1)

        try:
            with request.urlopen(req, timeout=timeout_sec) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                parsed: Any
                try:
                    parsed = json.loads(raw) if raw else {}
                except Exception:
                    parsed = raw
                status = int(getattr(resp, "status", 200))
                return 200 <= status < 300, status, parsed, None
        except error.HTTPError as e:
            raw = ""
            try:
                raw = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            return False, int(e.code), raw, f"http_error:{e.code}"
        except Exception as e:
            return False, None, None, f"network_error:{e}"
