"""
Phase 15 测试：PD 解耦（Disaggregated Prefill/Decode）

测试覆盖：
1. test_kv_transfer_payload        - KVPayload 序列化/反序列化
2. test_kv_transfer_queue          - KVSender/KVReceiver 通过 Queue 传输
3. test_extract_kv_from_past       - extract_kv_from_past 形状正确
4. test_rebuild_dynamic_cache      - _rebuild_dynamic_cache 形状正确
5. test_pd_engine_dry_run_single   - dry_run 单请求端到端
6. test_pd_engine_dry_run_batch    - dry_run 多请求端到端
7. test_worker_dry_run             - PrefillWorker + DecodeWorker 直接测试
"""

from __future__ import annotations

from multiprocessing import Queue

import torch
from transformers import DynamicCache

from mini_infer.core.config import EngineConfig
from mini_infer.cache.kv_transfer import (
    KVPayload,
    KVReceiver,
    KVSender,
    extract_kv_from_past,
    measure_kv_size_bytes,
)
from mini_infer.runtime.pd_engine import PDEngine
from mini_infer.runtime.pd_worker import (
    DecodeResult,
    PrefillRequest,
    _rebuild_dynamic_cache,
    run_decode_worker,
    run_prefill_worker,
)


def _dry_config() -> EngineConfig:
    return EngineConfig(model_name="test", dry_run=True, block_size=256)


# ──────────────────────────────────────────────────────────────────────────────
# 1. KVPayload 基本结构
# ──────────────────────────────────────────────────────────────────────────────

def test_kv_transfer_payload():
    k = torch.randn(4, 2, 8)  # [seq, heads, dim]
    v = torch.randn(4, 2, 8)
    payload = KVPayload(
        request_id="req-1",
        kv_layers=[(k, v), (k.clone(), v.clone())],
        first_token_id=42,
        prompt_len=4,
        max_new_tokens=16,
    )
    assert payload.request_id == "req-1"
    assert len(payload.kv_layers) == 2
    assert payload.first_token_id == 42
    size = measure_kv_size_bytes(payload.kv_layers)
    assert size == 2 * 2 * (4 * 2 * 8 * 4)  # 2 layers × 2 tensors × elements × float32 bytes


# ──────────────────────────────────────────────────────────────────────────────
# 2. KVSender / KVReceiver
# ──────────────────────────────────────────────────────────────────────────────

def test_kv_transfer_queue():
    q: Queue = Queue()
    sender = KVSender(q)
    receiver = KVReceiver(q)

    payload = KVPayload(request_id="req-2", first_token_id=7, prompt_len=3)
    sender.send(payload)
    received = receiver.recv(timeout=5.0)

    assert received.request_id == "req-2"
    assert received.first_token_id == 7


# ──────────────────────────────────────────────────────────────────────────────
# 3. extract_kv_from_past
# ──────────────────────────────────────────────────────────────────────────────

def test_extract_kv_from_past():
    # 构造一个 2 层 DynamicCache，batch=1, heads=2, seq=4, dim=8
    cache = DynamicCache()
    for i in range(2):
        k = torch.randn(1, 2, 4, 8)
        v = torch.randn(1, 2, 4, 8)
        cache.update(k, v, i)

    kv_layers = extract_kv_from_past(cache, seq_len=4)
    assert len(kv_layers) == 2
    k0, v0 = kv_layers[0]
    # 期望 shape: [seq_len, num_heads, head_dim]
    assert k0.shape == (4, 2, 8), f"expected (4,2,8), got {k0.shape}"
    assert v0.shape == (4, 2, 8)


# ──────────────────────────────────────────────────────────────────────────────
# 4. _rebuild_dynamic_cache
# ──────────────────────────────────────────────────────────────────────────────

def test_rebuild_dynamic_cache():
    kv_layers = [
        (torch.randn(4, 2, 8), torch.randn(4, 2, 8)),
        (torch.randn(4, 2, 8), torch.randn(4, 2, 8)),
    ]
    cache = _rebuild_dynamic_cache(kv_layers, device="cpu")
    assert len(cache.key_cache) == 2
    # 期望 shape: [1, num_heads, seq_len, head_dim]
    assert cache.key_cache[0].shape == (1, 2, 4, 8), f"got {cache.key_cache[0].shape}"


# ──────────────────────────────────────────────────────────────────────────────
# 5. PDEngine dry_run 单请求
# ──────────────────────────────────────────────────────────────────────────────

def test_pd_engine_dry_run_single():
    config = _dry_config()
    with PDEngine(config) as engine:
        results = engine.generate(["hello world"], max_new_tokens=4, timeout=30.0)
    assert len(results) == 1
    assert isinstance(results[0], str)
    assert len(results[0]) > 0


# ──────────────────────────────────────────────────────────────────────────────
# 6. PDEngine dry_run 多请求
# ──────────────────────────────────────────────────────────────────────────────

def test_pd_engine_dry_run_batch():
    config = _dry_config()
    prompts = ["prompt A", "prompt B", "prompt C"]
    with PDEngine(config) as engine:
        results = engine.generate(prompts, max_new_tokens=4, timeout=30.0)
    assert len(results) == 3
    for r in results:
        assert isinstance(r, str)
        assert len(r) > 0


# ──────────────────────────────────────────────────────────────────────────────
# 7. PrefillWorker + DecodeWorker 直接测试（dry_run）
# ──────────────────────────────────────────────────────────────────────────────

def test_worker_dry_run():
    """直接启动两个 worker 进程，验证消息流转正确。"""
    import multiprocessing as mp

    config = _dry_config()
    req_queue: Queue = mp.Queue()
    kv_queue: Queue = mp.Queue()
    result_queue: Queue = mp.Queue()
    stop = mp.Event()

    prefill_proc = mp.Process(
        target=run_prefill_worker,
        args=(config, req_queue, kv_queue, stop),
        daemon=True,
    )
    decode_proc = mp.Process(
        target=run_decode_worker,
        args=(config, kv_queue, result_queue, stop),
        daemon=True,
    )
    prefill_proc.start()
    decode_proc.start()

    req_queue.put(PrefillRequest(
        request_id="test-req-1",
        prompt="hello",
        max_new_tokens=3,
    ))

    result: DecodeResult = result_queue.get(timeout=30.0)

    stop.set()
    prefill_proc.join(timeout=5.0)
    decode_proc.join(timeout=5.0)

    assert result.request_id == "test-req-1"
    assert isinstance(result.output_text, str)
    assert len(result.output_text) > 0
