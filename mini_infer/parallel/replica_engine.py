"""
Phase 4 Replica 双卡引擎。

在两块 GPU 上各运行一个独立的 LLMEngine 实例，通过 ThreadPoolExecutor 并发执行：
- 偶数索引请求分发到 engines[0]（config_0.device，通常 cuda:0）
- 奇数索引请求分发到 engines[1]（config_1.device，通常 cuda:1）
- 两个引擎并发执行后合并结果，保持原始请求顺序

策略：数据并行（Replica）
- 显存消耗：每张卡一个完整模型（对于 Qwen2.5-7B: 2 × ~15.7 GB）
- 吞吐：理论约 2×（两卡独立并发）
- 单请求延迟：不变（与单卡相同）
- 适用场景：单卡能放下完整模型，追求高吞吐
"""

from concurrent.futures import ThreadPoolExecutor

from ..core.config import EngineConfig
from ..runtime.engine import LLMEngine


class ReplicaEngine:
    """
    双卡 Replica 推理引擎。

    两块 GPU 各运行一个完整的 LLMEngine，请求按 round-robin 分发并并发执行。
    两个 EngineConfig 的模型架构参数必须一致，device 字段必须不同（如 cuda:0 和 cuda:1）。
    """

    def __init__(self, config_0: EngineConfig, config_1: EngineConfig) -> None:
        if config_0.device == config_1.device:
            raise ValueError(
                f"ReplicaEngine 要求两个 config 使用不同的设备，"
                f"但都指定了 '{config_0.device}'。"
            )
        self.engines = [LLMEngine(config_0), LLMEngine(config_1)]

    def generate(self, prompts: list[str], max_new_tokens: int = 128) -> list[str]:
        """
        批量推理。
        偶数索引请求到 engines[0]，奇数索引到 engines[1]，并发执行后按原始顺序合并返回。
        """
        # round-robin 分组，保留原始索引以便合并时恢复顺序
        group_0 = [(i, p) for i, p in enumerate(prompts) if i % 2 == 0]
        group_1 = [(i, p) for i, p in enumerate(prompts) if i % 2 == 1]

        results: dict[int, str] = {}

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_0 = executor.submit(
                self._run_engine, 0, [p for _, p in group_0], max_new_tokens
            )
            future_1 = executor.submit(
                self._run_engine, 1, [p for _, p in group_1], max_new_tokens
            )
            outputs_0 = future_0.result(timeout=300)
            outputs_1 = future_1.result(timeout=300)

        for (orig_idx, _), out in zip(group_0, outputs_0):
            results[orig_idx] = out
        for (orig_idx, _), out in zip(group_1, outputs_1):
            results[orig_idx] = out

        return [results[i] for i in range(len(prompts))]

    def _run_engine(
        self,
        engine_idx: int,
        prompts: list[str],
        max_new_tokens: int,
    ) -> list[str]:
        """在指定引擎上运行推理，prompts 为空时直接返回空列表。"""
        if not prompts:
            return []
        return self.engines[engine_idx].generate(prompts, max_new_tokens)
