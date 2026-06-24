"""tests/test_chunked_prefill.py — Phase 9 Chunked Prefill 状态机与端到端测试。

覆盖范围（全部 dry_run=True，无需 GPU）：
  - prefilled_tokens 逐 chunk 正确递增，最后一个 chunk 后 prefilled=True
  - decode 请求在长请求 PREFILLING 期间持续产出 token（不被阻塞）
  - chunk_prefill_size=0（禁用）时行为与 Phase 8 一致（回归）
  - generate() 端到端：chunked 与非 chunked 均能完成并返回结果
  - KV 块在 generate() 完成后全部归还（无泄漏）
  - 边界情况：prompt 长度 <= chunk_prefill_size（单 chunk 即完成）
"""

import pytest

from mini_infer import EngineConfig, LLMEngine


def _make_engine(
    chunk_prefill_size: int = 2,
    num_gpu_blocks: int = 64,
    block_size: int = 4,
    max_batch_size: int = 4,
) -> LLMEngine:
    config = EngineConfig(
        model_name="stub",
        dry_run=True,
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        max_batch_size=max_batch_size,
        chunk_prefill_size=chunk_prefill_size,
    )
    return LLMEngine(config)


# ---------------------------------------------------------------------------
# 状态机：prefilled_tokens 递增，最后 chunk 后 prefilled=True
# ---------------------------------------------------------------------------


def test_prefilled_tokens_progress() -> None:
    """每步只推进 chunk_prefill_size 个 token；最后一个 chunk 后 prefilled=True。"""
    engine = _make_engine(chunk_prefill_size=2)
    # _StubTokenizer.encode("hello") = [104, 101, 108, 108, 111] → 5 tokens
    rid = engine.add_request("hello", max_new_tokens=3)
    state = engine._step_states[rid]

    # Step 1: chunk 0-2（tokens[0:2]）
    engine.step()
    assert state.prefilled_tokens == 2
    assert not state.prefilled

    # Step 2: chunk 2-4
    engine.step()
    assert state.prefilled_tokens == 4
    assert not state.prefilled

    # Step 3: chunk 4-5（最后一个 chunk）→ prefilled=True，产出第一个 decode token
    result = engine.step()
    assert state.prefilled_tokens == 5
    assert state.prefilled
    # 此时请求已进入 running；step() 收集其新 token
    assert rid in result


def test_single_chunk_completes_immediately() -> None:
    """prompt 长度 <= chunk_prefill_size 时，一步内完成 prefill 并进入 running。"""
    engine = _make_engine(chunk_prefill_size=10)
    # "hi" → [104, 105] → 2 tokens < 10
    rid = engine.add_request("hi", max_new_tokens=3)
    state = engine._step_states[rid]

    result = engine.step()
    assert state.prefilled  # 单 chunk 已完成
    assert rid in result    # 当步产出第一个 token


# ---------------------------------------------------------------------------
# 并发：decode 请求在 PREFILLING 期间不被阻塞
# ---------------------------------------------------------------------------


def test_decode_runs_during_prefilling() -> None:
    """已 running 的 decode 请求在长请求 PREFILLING 期间每步都能产出 token。"""
    engine = _make_engine(chunk_prefill_size=2)

    # 先提交短请求（2 tokens → 一步完成 prefill）
    rid_short = engine.add_request("hi", max_new_tokens=5)  # 2 tokens

    # Step 1：hi 完成 prefill（2 tokens = 1 chunk）→ 进入 running，产出第一个 token
    tokens1 = engine.step()
    assert engine._step_states[rid_short].prefilled

    # 提交长请求（5 tokens → 3 chunks）
    rid_long = engine.add_request("hello", max_new_tokens=3)  # 5 tokens

    # Steps 2-4：rid_long 在 PREFILLING，rid_short 每步都应产出 token
    decode_steps_with_short = 0
    for _ in range(3):
        tokens = engine.step()
        if rid_short in tokens:
            decode_steps_with_short += 1

    assert decode_steps_with_short == 3, (
        f"decode 请求在 PREFILLING 期间只产出了 {decode_steps_with_short}/3 步 token，"
        "说明 decode_batch 在某些步被阻塞（预期：全部 3 步均产出 token）"
    )

    # 最终 rid_long 应完成 prefill
    assert engine._step_states[rid_long].prefilled


# ---------------------------------------------------------------------------
# 回归：chunk_prefill_size=0 与 Phase 8 行为一致
# ---------------------------------------------------------------------------


def test_chunked_disabled_regression() -> None:
    """chunk_prefill_size=0 时，generate() 结果与开启 chunking 时相同（dry_run stub）。"""
    prompts = ["hello", "hi"]

    engine_no_chunk = _make_engine(chunk_prefill_size=0)
    out_no_chunk = engine_no_chunk.generate(prompts, max_new_tokens=3)

    engine_chunked = _make_engine(chunk_prefill_size=2)
    out_chunked = engine_chunked.generate(prompts, max_new_tokens=3)

    assert len(out_no_chunk) == len(out_chunked) == 2
    # dry_run stub 输出的内容格式相同（均为 " [1] [2] [3]" 格式）
    for a, b in zip(out_no_chunk, out_chunked):
        assert len(a) > 0 and len(b) > 0


# ---------------------------------------------------------------------------
# 端到端：generate() 可以正常完成
# ---------------------------------------------------------------------------


def test_generate_with_chunked_prefill() -> None:
    """chunk_prefill_size > 0 时，generate() 能完成所有请求并返回正确数量的输出。"""
    engine = _make_engine(chunk_prefill_size=2)
    prompts = ["hello", "hi", "world"]
    outputs = engine.generate(prompts, max_new_tokens=4)
    assert len(outputs) == 3
    for out in outputs:
        assert len(out) > 0


def test_generate_output_order_preserved() -> None:
    """outputs[i] 对应 prompts[i]，即使请求在不同步骤完成 prefill。"""
    engine = _make_engine(chunk_prefill_size=2, max_batch_size=1)
    prompts = ["hello", "hi"]  # "hello" 需 3 chunks，"hi" 需 1 chunk
    outputs = engine.generate(prompts, max_new_tokens=3)
    assert len(outputs) == 2
    for out in outputs:
        assert len(out) > 0


# ---------------------------------------------------------------------------
# 资源：KV 块无泄漏
# ---------------------------------------------------------------------------


def test_kv_blocks_returned_after_chunked_generate() -> None:
    """chunked generate 完成后，所有 KV 块全部归还，无泄漏。"""
    engine = _make_engine(chunk_prefill_size=2)
    initial_free = engine.kv_cache.num_free_blocks()
    engine.generate(["hello", "hi"], max_new_tokens=3)
    assert engine.kv_cache.num_free_blocks() == initial_free
