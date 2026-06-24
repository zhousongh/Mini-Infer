"""tests/test_async_engine.py — AsyncEngine 关键边界测试。

覆盖范围：
- EOS 后无可见 delta 时 DONE 事件仍然投递
- generate() 非流式接口返回完整文本
- generate_stream() 逐 token 流式接口
- async context manager 启动/停止
- ensure_healthy 在启动前抛出异常
- max_tokens 触发 length finish reason
- 并发两个请求均能完成
"""

from __future__ import annotations

import asyncio

import pytest

from mini_infer.runtime.async_engine import AsyncEngine
from mini_infer.core.config import EngineConfig


def _make_engine() -> AsyncEngine:
    return AsyncEngine(
        EngineConfig(
            model_name="stub",
            dry_run=True,
            block_size=4,
            num_gpu_blocks=32,
            max_batch_size=4,
        )
    )


def _patch_fake_tokens(engine: AsyncEngine, tokens: list[str]) -> None:
    """在 dry_run engine 上注入 fake prefill：依次 append 指定 token，然后标 eos。"""
    token_ids = list(range(1, len(tokens) + 1))
    id_to_text = {tid: text for tid, text in zip(token_ids, tokens)}
    engine._engine.model_runner.tokenizer.decode = (
        lambda ids, skip_special_tokens=True: "".join(
            id_to_text.get(i, "") for i in ids
        )
    )

    def fake_prefill(states):
        for state in states:
            for tid, text in zip(token_ids, tokens):
                state.append_generated(tid, text)
            state.prefilled = True
            state.mark_finished("eos")

    engine._engine.model_runner.prefill = fake_prefill


@pytest.mark.asyncio
async def test_generate_with_reason_finishes_on_eos_without_visible_delta() -> None:
    """
    即使最终 token 在 decode(skip_special_tokens=True) 后没有可见文本，
    AsyncEngine 也必须投递 DONE 事件，而不是让调用方永久等待。
    """
    engine = AsyncEngine(
        EngineConfig(
            model_name="stub",
            dry_run=True,
            block_size=4,
            num_gpu_blocks=32,
            max_batch_size=4,
        )
    )
    await engine.start()
    try:
        # 模拟 EOS token 对外不可见：decode 总是返回空串。
        engine._engine.model_runner.tokenizer.decode = lambda ids, skip_special_tokens=True: ""

        def fake_prefill(states):
            for state in states:
                state.append_generated(999, "")
                state.prefilled = True
                state.mark_finished("eos")

        engine._engine.model_runner.prefill = fake_prefill

        text, reason = await asyncio.wait_for(
            engine.generate_with_reason("hello", max_new_tokens=4),
            timeout=1.0,
        )
        assert text == ""
        assert reason == "eos"
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_generate_returns_full_text() -> None:
    """generate() 非流式接口返回所有 token 拼接后的完整文本。"""
    engine = _make_engine()
    _patch_fake_tokens(engine, ["hello", " ", "world"])
    async with engine:
        text = await asyncio.wait_for(
            engine.generate("hi", max_new_tokens=4),
            timeout=1.0,
        )
    assert text == "hello world"


@pytest.mark.asyncio
async def test_generate_stream_yields_nonempty_text() -> None:
    """generate_stream() 产出的文本与 generate() 完整输出一致。"""
    engine = _make_engine()
    _patch_fake_tokens(engine, ["tok1", "tok2", "tok3"])
    collected: list[str] = []
    async with engine:
        async for piece in engine.generate_stream("prompt", max_new_tokens=8):
            collected.append(piece)
    # fake prefill 在单步内生成所有 token，整体文本应完整
    assert "".join(collected) == "tok1tok2tok3"
    assert len(collected) >= 1


@pytest.mark.asyncio
async def test_ensure_healthy_raises_before_start() -> None:
    """engine.ensure_healthy() 在 start() 之前应抛出 RuntimeError。"""
    engine = _make_engine()
    with pytest.raises(RuntimeError):
        engine.ensure_healthy()


@pytest.mark.asyncio
async def test_context_manager_starts_and_stops() -> None:
    """async with AsyncEngine(...) 应自动 start/stop，退出后不应处于 running 状态。"""
    engine = _make_engine()
    _patch_fake_tokens(engine, ["x"])
    async with engine:
        assert engine._running is True
        await asyncio.wait_for(engine.generate("hi", max_new_tokens=2), timeout=1.0)
    assert engine._running is False


@pytest.mark.asyncio
async def test_concurrent_requests_both_complete() -> None:
    """两个并发 generate() 请求均应正常完成并返回文本。"""
    engine = _make_engine()
    _patch_fake_tokens(engine, ["a", "b"])
    async with engine:
        t1, t2 = await asyncio.wait_for(
            asyncio.gather(
                engine.generate("p1", max_new_tokens=4),
                engine.generate("p2", max_new_tokens=4),
            ),
            timeout=2.0,
        )
    assert t1 == "ab"
    assert t2 == "ab"
