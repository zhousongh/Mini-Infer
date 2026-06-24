"""
Phase 8 AsyncEngine：将同步 LLMEngine 包装为异步接口。

核心设计：
  - 后台线程持续调用 engine.step()，不等待 HTTP 请求凑齐
  - 每个 HTTP 请求通过 add_request() 加入等待队列，通过 asyncio.Queue 接收 token
  - 多个并发 HTTP 请求的 prompt 被 continuous batching 合并到同一 decode_batch

正确架构（解决"每请求独立调用 generate() 无法 continuous batching"的问题）：

  ❌ 错误：HTTP 请求 A → 线程1 → engine.generate(["A"])  ← 独占 GPU
           HTTP 请求 B → 线程2 → engine.generate(["B"])  ← 等待线程1

  ✅ 正确：HTTP 请求 A ─┐
           HTTP 请求 B ─┤─→ asyncio.Queue ─→ 后台 step loop ─→ decode_batch([A, B])
           HTTP 请求 C ─┘                      ↓ token 分发到各请求的 asyncio.Queue

并发设计要点：
  - 后台线程通过 loop.call_soon_threadsafe(queue.put_nowait, token) 跨线程投递 token
  - 前台 async generator 通过 await queue.get() 接收 token
  - 后台线程 sleep(1ms) 避免空转时的 CPU 浪费
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import AsyncGenerator
from uuid import uuid4

from ..core.config import EngineConfig
from .engine import LLMEngine

# 完成哨兵（不使用 None，避免与实际 token 冲突）
_DONE = object()


@dataclass(slots=True)
class _TokenEvent:
    text: str


@dataclass(slots=True)
class _DoneEvent:
    finish_reason: str | None


class AsyncEngine:
    """
    LLMEngine 的异步包装器。

    用法：
        engine = AsyncEngine(config)
        await engine.start()
        async for token in engine.generate_stream("hello"):
            print(token, end="", flush=True)
        await engine.stop()

    或使用 async context manager：
        async with AsyncEngine(config) as engine:
            result = await engine.generate("hello")
    """

    def __init__(self, config: EngineConfig) -> None:
        self._engine = LLMEngine(config)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._engine_lock = threading.Lock()
        self._running = False
        self._loop_error: Exception | None = None
        # request_id → asyncio.Queue（由后台线程投递 token，由 async generator 消费）
        self._token_queues: dict[str, asyncio.Queue] = {}

    async def start(self) -> None:
        """启动后台 step loop 线程。"""
        if self._running:
            raise RuntimeError("AsyncEngine 已启动，不能重复 start()。")
        self._loop = asyncio.get_running_loop()
        self._loop_error = None
        self._running = True
        self._thread = threading.Thread(target=self._engine_loop, daemon=True, name="mini-infer-step")
        self._thread.start()

    async def stop(self) -> None:
        """停止后台线程，等待其退出。"""
        if not self._running and self._thread is None:
            return
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._thread = None
        self._loop = None
        self._token_queues.clear()

    async def __aenter__(self) -> "AsyncEngine":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # 公开生成接口
    # ------------------------------------------------------------------

    def _raise_if_unhealthy(self) -> None:
        if self._loop_error is not None:
            raise RuntimeError("后台 step loop 已异常退出。") from self._loop_error
        if not self._running:
            raise RuntimeError("AsyncEngine 未启动。")

    def ensure_healthy(self) -> None:
        """供 HTTP 层在处理请求前做快速健康检查。"""
        self._raise_if_unhealthy()

    def kv_cache_stats(self) -> dict[str, object]:
        """返回 KV cache 状态快照，供 HTTP healthz/metrics 查询。"""
        self._raise_if_unhealthy()
        with self._engine_lock:
            return self._engine.kv_cache.stats()

    def scheduler_stats(self) -> dict[str, object]:
        """返回 scheduler 队列长度和累计事件计数。"""
        self._raise_if_unhealthy()
        with self._engine_lock:
            return self._engine.scheduler.stats()

    async def generate_stream_events(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        priority: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> AsyncGenerator[_TokenEvent | _DoneEvent, None]:
        """逐事件异步 yield；token 事件之后会收到一个完成事件。"""

        self._raise_if_unhealthy()

        rid = str(uuid4())
        queue: asyncio.Queue = asyncio.Queue()
        self._token_queues[rid] = queue  # 注册在 add_request 之前
        try:
            with self._engine_lock:
                self._raise_if_unhealthy()
                self._engine.add_request(
                    prompt,
                    max_new_tokens,
                    priority,
                    request_id=rid,
                    temperature=temperature,
                    top_p=top_p,
                )
            while True:
                try:
                    token = await asyncio.wait_for(queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # 长 prefill / 大 batch 下可能较久都没有新 token，定期做健康检查即可，
                    # 不对正常慢请求施加固定 60s 的硬超时。
                    self._raise_if_unhealthy()
                    continue
                if isinstance(token, _DoneEvent):
                    if self._loop_error is not None:
                        raise RuntimeError("后台 step loop 已异常退出。") from self._loop_error
                    yield token
                    return
                yield token  # type: ignore[misc]
        finally:
            self._token_queues.pop(rid, None)
            with self._engine_lock:
                self._engine.cancel_request(rid)

    async def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        priority: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> AsyncGenerator[str, None]:
        """逐增量文本异步 yield，直到生成完毕。"""
        async for event in self.generate_stream_events(
            prompt,
            max_new_tokens=max_new_tokens,
            priority=priority,
            temperature=temperature,
            top_p=top_p,
        ):
            if isinstance(event, _TokenEvent):
                yield event.text

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        priority: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """非流式版本：等待所有 token 后一次返回完整文本。"""
        text, _ = await self.generate_with_reason(
            prompt,
            max_new_tokens=max_new_tokens,
            priority=priority,
            temperature=temperature,
            top_p=top_p,
        )
        return text

    async def generate_with_reason(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        priority: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> tuple[str, str]:
        """非流式版本：返回完整文本与实际 finish_reason。"""
        parts: list[str] = []
        finish_reason = "stop"
        async for event in self.generate_stream_events(
            prompt,
            max_new_tokens=max_new_tokens,
            priority=priority,
            temperature=temperature,
            top_p=top_p,
        ):
            if isinstance(event, _TokenEvent):
                parts.append(event.text)
            else:
                finish_reason = event.finish_reason or "stop"
        return "".join(parts), finish_reason

    @property
    def tokenizer(self):
        """暴露底层 tokenizer，供 server.py 计算 token 数量。"""
        return self._engine.model_runner.tokenizer

    def format_prompt(self, messages: list) -> str:
        """
        将 ChatMessage 列表转换为模型输入字符串。

        优先使用 tokenizer.apply_chat_template（Qwen 等模型有正式格式）；
        dry_run / stub tokenizer 时退化为简单拼接。
        """
        tokenizer = self._engine.model_runner.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                [{"role": m.role, "content": m.content} for m in messages],
                tokenize=False,
                add_generation_prompt=True,
            )
        # Fallback：简单格式，仅用于 dry_run 测试
        parts = [f"{m.role}: {m.content}" for m in messages]
        parts.append("assistant:")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # 后台 step loop（在独立线程中运行）
    # ------------------------------------------------------------------

    def _engine_loop(self) -> None:
        while self._running:
            with self._engine_lock:
                has_work = self._engine.has_unfinished_requests()
            if has_work:
                try:
                    with self._engine_lock:
                        new_tokens = self._engine.step()
                        tracked_rids = list(self._token_queues)
                        finished = {
                            rid: self._engine.get_finish_reason(rid)
                            for rid in tracked_rids
                            if self._engine.is_finished(rid)
                        }
                    for rid, tokens in new_tokens.items():
                        for text in tokens:
                            self._put(rid, _TokenEvent(text))
                    # 即使本步没有可见文本增量，也必须为已完成请求投递 DONE 哨兵。
                    for rid, finish_reason in finished.items():
                        self._put(rid, _DoneEvent(finish_reason))
                        with self._engine_lock:
                            self._engine.cleanup_request(rid)
                except Exception as exc:
                    self._loop_error = exc
                    self._running = False
                    import traceback
                    traceback.print_exc()
                    # 错误时通知所有等待中的消费者退出，并清理追踪表
                    for rid in list(self._token_queues):
                        self._put(rid, _DoneEvent("error"))
                    with self._engine_lock:
                        for rid in list(self._token_queues):
                            self._engine.cancel_request(rid)
                    break
            else:
                time.sleep(0.001)  # 无请求时短暂休眠，避免空转

    def _put(self, rid: str, item: object) -> None:
        """线程安全地向 asyncio.Queue 投递 item。"""
        queue = self._token_queues.get(rid)
        if queue is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(queue.put_nowait, item)
