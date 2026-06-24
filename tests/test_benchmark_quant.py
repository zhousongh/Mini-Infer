"""Phase 16 benchmark 口径测试。

覆盖：
- token match / sequence exact 指标定义
- 固定 12 条 prompt 的分批执行口径
- 模型几何参数自动推导与 build_runner 接线
- mixed fallback compute note
- linear sweep 的层选择与统计窗口
- linear sweep 的 dtype 对齐
- run_benchmark 的异常安全清理
- benchmark CLI compare 主流程参数传递
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from mini_infer.modeling.quantization import QuantLinear
from mini_infer.core.request import RequestState


_MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_quant.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_quant", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
benchmark_quant = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark_quant)


def test_compute_match_metrics_exact_match() -> None:
    metrics = benchmark_quant.compute_match_metrics(
        [[1, 2, 3], [4, 5]],
        [[1, 2, 3], [4, 5]],
    )
    assert metrics["token_match_rate"] == 1.0
    assert metrics["sequence_exact_rate"] == 1.0


def test_compute_match_metrics_counts_length_mismatch_as_mismatch() -> None:
    metrics = benchmark_quant.compute_match_metrics(
        [[1, 2, 3], [7, 8]],
        [[1, 9], [7, 8, 10]],
    )
    assert metrics["token_matches"] == 3.0
    assert metrics["token_total"] == 6.0
    assert metrics["token_match_rate"] == 0.5
    assert metrics["sequence_exact_rate"] == 0.0


def test_build_prompt_batches_covers_all_prompts() -> None:
    batches = benchmark_quant.build_prompt_batches(batch_size=5)
    assert [len(batch) for batch in batches] == [5, 5, 2]
    assert sum(len(batch) for batch in batches) == len(benchmark_quant.PROMPTS)


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = nn.Linear(8, 8)
        self.gate_proj = nn.Linear(8, 16)
        self.other = nn.Linear(8, 8)


def test_find_benchmark_linear_prefers_gate_proj_for_fp16() -> None:
    name, module = benchmark_quant.find_benchmark_linear(_ToyModel(), "fp16")
    assert name.endswith("gate_proj")
    assert isinstance(module, nn.Linear)


def test_find_benchmark_linear_prefers_quantized_gate_proj_for_w8a8() -> None:
    model = _ToyModel()
    model.gate_proj = QuantLinear(model.gate_proj.weight.data, model.gate_proj.bias.data)
    name, module = benchmark_quant.find_benchmark_linear(model, "w8a8")
    assert name.endswith("gate_proj")
    assert isinstance(module, QuantLinear)


def test_build_linear_sweep_rows() -> None:
    rows = benchmark_quant.build_linear_sweep_rows(max_rows=20)
    assert rows == [1, 2, 4, 8, 16]


def test_build_quant_compute_note_marks_mixed_fallback() -> None:
    note = benchmark_quant.build_quant_compute_note(
        {
            "weight_storage": "int8_per_channel",
            "int_mm_activation_granularity": "per_row",
            "fallback_activation_granularity": "fp32",
            "int_mm_min_rows": 17,
            "fallback_compute": "float_activation_x_dequant_weight",
        },
        {
            "int_mm_calls": 0,
            "fallback_calls": 5,
            "int_mm_rows": 0,
            "fallback_rows": 20,
        },
    )

    assert note is not None
    assert "weight_storage=int8_per_channel" in note
    assert "fallback_compute=float_activation_x_dequant_weight" in note
    assert "mixed fallback compute" in note


def test_infer_model_geometry_reads_hf_config(monkeypatch) -> None:
    fake_cfg = SimpleNamespace(
        num_hidden_layers=28,
        num_key_value_heads=2,
        num_attention_heads=12,
        hidden_size=1536,
    )
    fake_auto_config = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: fake_cfg
    )
    monkeypatch.setattr(benchmark_quant, "AutoConfig", fake_auto_config)

    geometry = benchmark_quant.infer_model_geometry("dummy-model")

    assert geometry == {
        "num_hidden_layers": 28,
        "num_kv_heads": 2,
        "head_dim": 128,
    }


def test_build_runner_uses_inferred_geometry(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        benchmark_quant,
        "infer_model_geometry",
        lambda model_path: {
            "num_hidden_layers": 30,
            "num_kv_heads": 4,
            "head_dim": 96,
        },
    )

    class _FakeKVCacheManager:
        def __init__(self, cfg):
            captured["kv_cfg"] = cfg

    class _FakeModelRunner:
        def __init__(self, cfg, kv_cache):
            captured["runner_cfg"] = cfg
            captured["runner_kv"] = kv_cache

    monkeypatch.setattr(benchmark_quant, "KVCacheManager", _FakeKVCacheManager)
    monkeypatch.setattr(benchmark_quant, "ModelRunner", _FakeModelRunner)

    cfg, kv_cache, runner = benchmark_quant.build_runner(
        model_path="dummy-model",
        mode="w8a8",
        batch_size=4,
        num_gpu_blocks=123,
    )

    assert cfg.num_hidden_layers == 30
    assert cfg.num_kv_heads == 4
    assert cfg.head_dim == 96
    assert cfg.max_batch_size == 4
    assert cfg.num_gpu_blocks == 123
    assert cfg.quant_mode == "w8a8"
    assert captured["kv_cfg"] is cfg
    assert captured["runner_cfg"] is cfg
    assert captured["runner_kv"] is kv_cache
    assert isinstance(runner, _FakeModelRunner)


class _ToyQuantModel(nn.Module):
    def __init__(self):
        super().__init__()
        linear = nn.Linear(8, 16)
        self.gate_proj = QuantLinear(linear.weight.data, linear.bias.data)


def test_run_linear_sweep_stats_exclude_warmup() -> None:
    runner = SimpleNamespace(
        model=_ToyQuantModel(),
        config=SimpleNamespace(device="cpu", dtype="float32"),
    )

    result = benchmark_quant.run_linear_sweep(
        runner,
        "w8a8",
        warmup_iters=2,
        bench_iters=3,
        sweep_rows=[4],
    )

    row = result["rows"][0]
    assert row["quant_stats"]["fallback_calls"] == 3
    assert row["quant_stats"]["fallback_rows"] == 12


def test_run_linear_sweep_uses_bfloat16_dtype(monkeypatch) -> None:
    seen_dtypes: list[torch.dtype] = []
    orig_randn = benchmark_quant.torch.randn

    def fake_randn(*args, **kwargs):
        seen_dtypes.append(kwargs["dtype"])
        return orig_randn(*args, **kwargs)

    monkeypatch.setattr(benchmark_quant.torch, "randn", fake_randn)
    runner = SimpleNamespace(
        model=_ToyQuantModel(),
        config=SimpleNamespace(device="cpu", dtype="bfloat16"),
    )

    benchmark_quant.run_linear_sweep(
        runner,
        "w8a8",
        warmup_iters=0,
        bench_iters=1,
        sweep_rows=[4],
    )

    assert seen_dtypes == [torch.bfloat16]


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [len(text)]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return ",".join(str(token_id) for token_id in token_ids)


class _FakeRunner:
    def __init__(self):
        self.model = nn.Linear(1, 1)
        self.config = SimpleNamespace(device="cpu", dtype="float32")
        self.tokenizer = _FakeTokenizer()

    def prefill(self, states: list[RequestState]) -> None:
        for state in states:
            state.append_generated(100, "")
            state.prefilled = True

    def decode_batch(self, states: list[RequestState]) -> None:
        for state in states:
            state.append_generated(101, "")
            state.mark_finished("length")


class _FakeKVCache:
    def init_request(self, state: RequestState) -> None:
        return None

    def free_request(self, state: RequestState) -> None:
        return None


def test_run_benchmark_uses_all_prompts_across_batches() -> None:
    result = benchmark_quant.run_benchmark(
        runner=_FakeRunner(),
        kv_cache=_FakeKVCache(),
        mode="fp16",
        batch_size=5,
        max_new_tokens=2,
        warmup_iters=0,
        bench_iters=1,
    )

    assert result["prompt_count"] == len(benchmark_quant.PROMPTS)
    assert result["num_prompt_batches"] == 3
    assert len(result["output_token_ids"]) == len(benchmark_quant.PROMPTS)
    assert result["quant_contract"] is None
    assert result["quant_compute_note"] is None


class _TrackingKVCache(_FakeKVCache):
    def __init__(self) -> None:
        self.init_ids: list[str] = []
        self.free_ids: list[str] = []

    def init_request(self, state: RequestState) -> None:
        self.init_ids.append(state.request.request_id)

    def free_request(self, state: RequestState) -> None:
        self.free_ids.append(state.request.request_id)


class _ExplodingRunner(_FakeRunner):
    def prefill(self, states: list[RequestState]) -> None:
        raise RuntimeError("prefill boom")


def test_run_benchmark_frees_kv_on_exception() -> None:
    kv_cache = _TrackingKVCache()

    with pytest.raises(RuntimeError, match="prefill boom"):
        benchmark_quant.run_benchmark(
            runner=_ExplodingRunner(),
            kv_cache=kv_cache,
            mode="fp16",
            batch_size=5,
            max_new_tokens=2,
            warmup_iters=0,
            bench_iters=1,
        )

    assert kv_cache.free_ids == kv_cache.init_ids


def test_main_compare_passes_reference_tokens(monkeypatch) -> None:
    calls: list[tuple[str, str, object]] = []

    def fake_build_runner(model_path, mode, batch_size, num_gpu_blocks):
        return object(), f"kv-{mode}", f"runner-{mode}"

    def fake_run_benchmark(
        runner,
        kv_cache,
        mode,
        batch_size,
        max_new_tokens,
        warmup_iters=2,
        bench_iters=5,
        reference_token_ids=None,
    ):
        calls.append(("run_benchmark", mode, reference_token_ids))
        return {"mode": mode, "output_token_ids": [[1, 2, 3]]}

    monkeypatch.setattr(benchmark_quant, "build_runner", fake_build_runner)
    monkeypatch.setattr(benchmark_quant, "run_benchmark", fake_run_benchmark)
    monkeypatch.setattr(
        benchmark_quant,
        "run_linear_sweep",
        lambda runner, mode, warmup_iters=10, bench_iters=50: {
            "mode": mode,
            "layer_name": "layer",
            "in_features": 8,
            "out_features": 16,
            "rows": [],
        },
    )
    monkeypatch.setattr(benchmark_quant, "print_result", lambda result: None)
    monkeypatch.setattr(benchmark_quant, "print_comparison", lambda fp16, w8a8: None)
    monkeypatch.setattr(
        benchmark_quant,
        "print_linear_sweep_comparison",
        lambda fp16, w8a8: None,
    )
    monkeypatch.setattr(benchmark_quant.gc, "collect", lambda: None)
    monkeypatch.setattr(benchmark_quant, "_sync_cuda", lambda: None)
    monkeypatch.setattr(benchmark_quant.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_quant.py",
            "--model",
            "dummy-model",
            "--compare",
            "--batch-size",
            "4",
            "--max-new-tokens",
            "8",
        ],
    )

    benchmark_quant.main()

    assert calls == [
        ("run_benchmark", "fp16", None),
        ("run_benchmark", "w8a8", [[1, 2, 3]]),
    ]


def test_format_linear_quant_stats_without_data() -> None:
    assert benchmark_quant.format_linear_quant_stats(None) == "N/A"
