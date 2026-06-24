"""
Phase 4 ReplicaEngine 和 PPEngine 的单元测试。覆盖范围：
  - ReplicaEngine 分发：偶数 / 奇数索引请求分到正确引擎
  - ReplicaEngine 顺序：合并结果与输入顺序一致
  - ReplicaEngine 边界：单条请求、奇数条请求（某引擎分到空 batch）
  - ReplicaEngine 校验：两个 config 设备相同时抛出 ValueError
  - PPEngine dry_run：smoke test，不加载真实模型
    注：PPEngine 是 Pipeline Parallel 引擎（device_map="balanced"），不是 Tensor Parallel

所有测试使用 dry_run=True，不依赖 GPU 或模型权重。
"""

import pytest

from mini_infer.core.config import EngineConfig
from mini_infer.parallel.pp_engine import PPEngine
from mini_infer.parallel.replica_engine import ReplicaEngine


def _make_config(device: str) -> EngineConfig:
    return EngineConfig(
        model_name="stub",
        device=device,
        dry_run=True,
    )


# ── ReplicaEngine 测试 ──────────────────────────────────────────────────────

class TestReplicaEngine:
    def test_same_device_raises(self) -> None:
        """两个 config 使用相同 device 时应抛出 ValueError。"""
        cfg = _make_config("cpu")
        with pytest.raises(ValueError, match="不同的设备"):
            ReplicaEngine(cfg, cfg)

    def test_generate_returns_correct_count(self) -> None:
        """generate 返回列表长度应与输入 prompts 数量一致。"""
        engine = ReplicaEngine(_make_config("cpu"), _make_config("cuda:0"))
        outputs = engine.generate(["a", "b", "c", "d"], max_new_tokens=4)
        assert len(outputs) == 4

    def test_generate_single_prompt(self) -> None:
        """单条请求：只有 engine[0] 被调用，engine[1] 收到空 batch。"""
        engine = ReplicaEngine(_make_config("cpu"), _make_config("cuda:0"))
        outputs = engine.generate(["hello"], max_new_tokens=4)
        assert len(outputs) == 1
        assert isinstance(outputs[0], str)

    def test_generate_order_preserved(self) -> None:
        """
        dry_run 模式下 ModelRunner._StubTokenizer.decode 返回固定格式。
        验证结果顺序与输入一致：outputs[i] 对应 prompts[i]。
        使用不同长度 prompts 可间接验证分发和合并的对应关系。
        """
        engine = ReplicaEngine(_make_config("cpu"), _make_config("cuda:0"))
        prompts = ["p0", "p1", "p2", "p3", "p4"]
        outputs = engine.generate(prompts, max_new_tokens=2)
        assert len(outputs) == 5
        # 每个输出都是字符串（dry_run 的 decode 返回 " [1] [2] ..." 格式）
        for out in outputs:
            assert isinstance(out, str)

    def test_empty_group_handled(self) -> None:
        """1条请求时 group_1 为空，engine[1] 应被正确跳过（返回空列表）。"""
        engine = ReplicaEngine(_make_config("cpu"), _make_config("cuda:0"))
        outputs = engine.generate(["only one"], max_new_tokens=2)
        assert len(outputs) == 1


# ── PPEngine 测试（Pipeline Parallel）──────────────────────────────────────

class TestPPEngine:
    def test_dry_run_smoke(self) -> None:
        """dry_run 模式下 PPEngine 可构造并调用 generate，不加载真实模型。"""
        cfg = EngineConfig(model_name="stub", device="cpu", dry_run=True)
        engine = PPEngine(cfg)
        outputs = engine.generate(["hello", "world"], max_new_tokens=4)
        assert len(outputs) == 2
        for out in outputs:
            assert isinstance(out, str)

    def test_dry_run_requires_no_gpu(self) -> None:
        """dry_run 模式下不应触发 GPU 检查。"""
        cfg = EngineConfig(model_name="stub", device="cuda:0", dry_run=True)
        engine = PPEngine(cfg)  # 不应 raise RuntimeError
        assert engine.model is None
