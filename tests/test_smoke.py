"""
这个文件覆盖引擎的最小 smoke test，使用 dry_run=True 在无 GPU 环境下验证 generate 主链路的状态流转。
Phase 2 新增：OOM 检测（请求所需块超过总量时应抛出 RuntimeError 而非死循环）。
"""

import pytest

from mini_infer import EngineConfig, LLMEngine


def test_engine_generate_smoke() -> None:
    config = EngineConfig(model_name="stub-model", max_batch_size=2, block_size=4, dry_run=True)
    engine = LLMEngine(config)
    outputs = engine.generate(["hello", "world"], max_new_tokens=2)

    assert len(outputs) == 2
    assert outputs[0] == " [1] [2]"
    assert outputs[1] == " [1] [2]"


def test_engine_oom_raises_not_loops() -> None:
    """当请求所需 KV 块超过总量时，应抛出 RuntimeError，不得死循环。"""
    # 设置极小的 block pool：2 块 × 4 slots = 8 token 容量
    # prompt "hello" = 5 tokens，max_new_tokens=10，合计需要 ceil(15/4)=4 块 > 2 块
    config = EngineConfig(
        model_name="stub-model",
        block_size=4,
        num_gpu_blocks=2,
        dry_run=True,
    )
    engine = LLMEngine(config)
    with pytest.raises(RuntimeError, match="KV 块"):
        engine.generate(["hello"], max_new_tokens=10)
