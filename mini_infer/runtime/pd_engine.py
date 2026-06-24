"""
Phase 15：PDEngine — PD 解耦协调层。

对外暴露与 LLMEngine 相同的 generate() 接口，内部启动两个子进程：
  - PrefillWorker：执行 prefill，提取 KV，发送 KVPayload
  - DecodeWorker：接收 KVPayload，执行 decode loop，返回结果

时序：
  generate(prompts)
    → 逐个发送 PrefillRequest 到 prefill_req_queue
    → PrefillWorker prefill → KVPayload → kv_queue
    → DecodeWorker decode → DecodeResult → result_queue
    → 收集所有结果，按原始顺序返回

计时分解（每个请求）：
  prefill_time：PrefillWorker 内部计时（写入 KVPayload.prefill_time）
  transfer_time：KVPayload 入队到 DecodeWorker 收到的时间差（由 PDEngine 估算）
  decode_time：DecodeWorker 内部计时（写入 DecodeResult.decode_time）
"""

from __future__ import annotations

import multiprocessing as _mp
import time
from uuid import uuid4

from ..core.config import EngineConfig
from .pd_worker import (
    DecodeResult,
    PrefillRequest,
    run_decode_worker,
    run_prefill_worker,
)

# CUDA 不兼容 fork，必须使用 spawn（Linux 默认 fork 会导致 CUDA 死锁）
_ctx = _mp.get_context("spawn")


class PDEngine:
    """
    PD 解耦推理引擎。

    用法：
        engine = PDEngine(config)
        results = engine.generate(["hello", "world"])
        engine.shutdown()

    或使用上下文管理器：
        with PDEngine(config) as engine:
            results = engine.generate(["hello"])
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._stop = _ctx.Event()

        # 进程间通信队列
        self._prefill_req_queue = _ctx.Queue()
        self._kv_queue = _ctx.Queue()
        self._result_queue = _ctx.Queue()

        # 启动子进程（spawn context：CUDA 安全）
        self._prefill_proc = _ctx.Process(
            target=run_prefill_worker,
            args=(config, self._prefill_req_queue, self._kv_queue, self._stop),
            daemon=True,
            name="PrefillWorker",
        )
        self._decode_proc = _ctx.Process(
            target=run_decode_worker,
            args=(config, self._kv_queue, self._result_queue, self._stop),
            daemon=True,
            name="DecodeWorker",
        )
        self._prefill_proc.start()
        self._decode_proc.start()

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: float = 120.0,
    ) -> list[str]:
        """
        批量生成，返回与输入 prompts 顺序一致的输出文本列表。

        注意：当前实现串行发送请求（一次一个），不支持真正的并发 batch。
        Phase 15 目标是验证架构正确性，不优化吞吐。
        """
        request_ids: list[str] = []

        for prompt in prompts:
            rid = str(uuid4())
            request_ids.append(rid)
            self._prefill_req_queue.put(PrefillRequest(
                request_id=rid,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            ))

        # 收集结果（按 request_id 对应）
        results: dict[str, DecodeResult] = {}
        deadline = time.perf_counter() + timeout

        while len(results) < len(prompts):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(
                    f"PDEngine.generate() 超时：已收到 {len(results)}/{len(prompts)} 个结果"
                )
            try:
                result: DecodeResult = self._result_queue.get(timeout=min(remaining, 5.0))
                results[result.request_id] = result
            except Exception:
                continue

        return [results[rid].output_text for rid in request_ids]

    def generate_with_timing(
        self,
        prompts: list[str],
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: float = 120.0,
    ) -> list[dict]:
        """
        与 generate() 相同，但额外返回每个请求的计时分解。

        返回：list of {output_text, prefill_time, decode_time, transfer_time, request_id}
        """
        request_ids: list[str] = []

        for prompt in prompts:
            rid = str(uuid4())
            request_ids.append(rid)
            self._prefill_req_queue.put(PrefillRequest(
                request_id=rid,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            ))

        results: dict[str, DecodeResult] = {}
        deadline = time.perf_counter() + timeout

        while len(results) < len(prompts):
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"超时：已收到 {len(results)}/{len(prompts)} 个结果")
            try:
                result: DecodeResult = self._result_queue.get(timeout=min(remaining, 5.0))
                results[result.request_id] = result
            except Exception:
                continue

        return [
            {
                "request_id": rid,
                "output_text": results[rid].output_text,
                "prefill_time": results[rid].prefill_time,
                "decode_time": results[rid].decode_time,
                "transfer_time": results[rid].transfer_time,
            }
            for rid in request_ids
        ]

    def shutdown(self) -> None:
        """停止两个 worker 进程。"""
        self._stop.set()
        self._prefill_proc.join(timeout=5.0)
        self._decode_proc.join(timeout=5.0)
        if self._prefill_proc.is_alive():
            self._prefill_proc.terminate()
        if self._decode_proc.is_alive():
            self._decode_proc.terminate()

    def __enter__(self) -> "PDEngine":
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()
