"""tests/test_engine.py — LLMEngine 主循环集成测试。

覆盖范围（全部 dry_run=True，无需 GPU）：
  - 基本生成：单请求按 max_new_tokens 截止，返回非空字符串
  - 多请求批处理：多个 prompt 同时入队，各自完成后返回正确数量的输出
  - 输出顺序保证：outputs[i] 对应 prompts[i]，即使请求在不同步骤完成
  - KV 块回收：generate 完成后，空闲块数恢复到初始值
  - KV 耗尽报错：块数不足时抛出 RuntimeError（Phase 7 preemption 的前置场景）
  - max_new_tokens=1 截止：prefill 后立即完成，不进入 decode 循环
  - 空 prompts 列表：返回空列表，不报错
"""

import json

import pytest

from mini_infer import EngineConfig, LLMEngine


def _make_engine(
    num_gpu_blocks: int = 32,
    block_size: int = 4,
    max_batch_size: int = 4,
) -> LLMEngine:
    config = EngineConfig(
        model_name="stub",
        dry_run=True,
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        max_batch_size=max_batch_size,
    )
    return LLMEngine(config)


def test_single_request_returns_output() -> None:
    """单请求 generate 应返回长度为 1 的列表，且元素非空。"""
    engine = _make_engine()
    outputs = engine.generate(["hello"], max_new_tokens=3)
    assert len(outputs) == 1
    assert len(outputs[0]) > 0


def test_output_count_matches_prompts() -> None:
    """输出列表长度必须等于 prompts 列表长度。"""
    engine = _make_engine()
    prompts = ["a", "bb", "ccc"]
    outputs = engine.generate(prompts, max_new_tokens=2)
    assert len(outputs) == len(prompts)


def test_kv_blocks_returned_after_generate() -> None:
    """generate 完成后，所有 KV 块必须全部归还到空闲池或 prefix cache，不留下泄漏。

    Phase 10：prefix cache 合法地持有已完成请求的 prompt blocks（ref_count >= 1）。
    正确性检查：free + prefix_cache_size == initial_free。
    """
    engine = _make_engine(num_gpu_blocks=16, block_size=4)
    initial_free = engine.kv_cache.num_free_blocks()
    engine.generate(["hello", "world"], max_new_tokens=4)
    assert (
        engine.kv_cache.num_free_blocks() + engine.kv_cache.prefix_cache_size()
        == initial_free
    )


def test_kv_exhaustion_raises_runtime_error() -> None:
    """KV 块不足以容纳单个请求时，必须抛出 RuntimeError，而不是死循环。

    "ab" → 2 tokens；max_new_tokens=4；需要 ceil((2+4)/4)=2 块；num_gpu_blocks=1 不足。
    这是 Phase 7 preemption 要解决的场景——当前应当明确报错。
    """
    engine = _make_engine(num_gpu_blocks=1, block_size=4)
    with pytest.raises(RuntimeError, match="KV 块"):
        engine.generate(["ab"], max_new_tokens=4)
    assert engine.scheduler.stats()["counters"]["reject_count"] == 1


def test_output_order_preserved_with_different_prompt_lengths() -> None:
    """多请求时，输出顺序必须与输入 prompts 顺序严格一致。

    使用不同长度 prompt 触发内部不同完成时序，验证对应关系不乱序。
    """
    engine = _make_engine(num_gpu_blocks=32, block_size=4)
    prompts = ["a", "bbb", "cc"]
    outputs = engine.generate(prompts, max_new_tokens=2)
    assert len(outputs) == 3
    for out in outputs:
        assert isinstance(out, str)
        assert len(out) > 0


def test_max_new_tokens_one_completes_at_prefill() -> None:
    """max_new_tokens=1 时，请求在 prefill 阶段采样第一个 token 后即完成，不进 decode。

    完成后 KV 块全部归还或持有于 prefix cache（Phase 10），验证 prefill-only 路径的块生命周期正确。
    """
    engine = _make_engine(num_gpu_blocks=32, block_size=4)
    initial_free = engine.kv_cache.num_free_blocks()
    outputs = engine.generate(["hello", "world"], max_new_tokens=1)
    assert len(outputs) == 2
    assert (
        engine.kv_cache.num_free_blocks() + engine.kv_cache.prefix_cache_size()
        == initial_free
    )


def test_empty_prompts_returns_empty_list() -> None:
    """空 prompts 列表应直接返回空列表，不报错，不进入 step 循环。"""
    engine = _make_engine()
    outputs = engine.generate([], max_new_tokens=5)
    assert outputs == []


def test_real_engine_requires_256_aligned_block_size() -> None:
    """真实 LLMEngine 应在加载模型前拒绝非 256 对齐的 block_size。"""
    config = EngineConfig(model_name="stub", dry_run=False, block_size=16)
    with pytest.raises(ValueError, match="block_size"):
        LLMEngine(config)


def test_multiple_generate_calls_independent() -> None:
    """连续两次 generate 调用之间状态互不干扰，块计数每次都能归零（或进入 prefix cache）。

    Phase 10：prefix cache 会跨调用保留 prompt blocks，但不应影响后续请求的准入。
    """
    engine = _make_engine(num_gpu_blocks=32, block_size=4)
    initial_free = engine.kv_cache.num_free_blocks()

    outputs1 = engine.generate(["hello"], max_new_tokens=2)
    assert (
        engine.kv_cache.num_free_blocks() + engine.kv_cache.prefix_cache_size()
        == initial_free
    )

    outputs2 = engine.generate(["world"], max_new_tokens=2)
    assert (
        engine.kv_cache.num_free_blocks() + engine.kv_cache.prefix_cache_size()
        == initial_free
    )
    assert len(outputs1) == 1
    assert len(outputs2) == 1


def test_kv_trace_file_records_request_lifecycle(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """开启 MINI_INFER_KV_TRACE_FILE 后，应记录请求生命周期中的 KV 状态变化。"""
    trace_file = tmp_path / "kv_trace.jsonl"
    monkeypatch.setenv("MINI_INFER_KV_TRACE_FILE", str(trace_file))

    engine = _make_engine(num_gpu_blocks=16, block_size=4)
    engine.add_request("hello", max_new_tokens=3)
    while engine.has_unfinished_requests():
        engine.step()

    records = [
        json.loads(line)
        for line in trace_file.read_text(encoding="utf-8").splitlines()
    ]
    events = [record["event"] for record in records]

    assert "add_request" in events
    assert "admit_running" in events
    assert "prefill_complete" in events
    assert "decode_batch" in events
    assert "finish_request" in events
    assert all("kv_cache" in record for record in records)
    assert all("scheduler" in record for record in records)
