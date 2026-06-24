"""Phase 17-21 benchmark 口径测试。

覆盖：
- TP / EP 通信量公式
- 参数量与 local shard 统计
- padded / packed / grouped bytes 统计与输出结构
- packed control-plane 指标与输出结构
- grouped local expert execution 的 compare / CLI 接线
- synthetic hidden state 构造
- benchmark dry-run 所需的参数构造 helper
- dense benchmark 计时窗口与 source device 同步口径
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


_MODULE_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "benchmark_moe.py"
_SPEC = importlib.util.spec_from_file_location("benchmark_moe", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
benchmark_moe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(benchmark_moe)


def test_compute_tp_bytes_per_layer() -> None:
    bytes_per_layer = benchmark_moe.compute_tp_bytes_per_layer(
        num_tokens=8,
        hidden_size=16,
        dtype_bytes=2,
    )
    assert bytes_per_layer == 512


def test_compute_ep_bytes_per_layer() -> None:
    bytes_per_layer = benchmark_moe.compute_ep_bytes_per_layer(
        num_tokens=8,
        hidden_size=16,
        top_k=2,
        dtype_bytes=2,
    )
    assert bytes_per_layer == 1024


def test_compute_ep_padded_bytes_per_layer() -> None:
    bytes_per_layer = benchmark_moe.compute_ep_padded_bytes_per_layer(
        num_tokens=8,
        hidden_size=16,
        top_k=2,
        dtype_bytes=2,
        ep_size=2,
    )
    assert bytes_per_layer == 2048


def test_compute_ep_packed_bytes_per_layer() -> None:
    bytes_per_layer = benchmark_moe.compute_ep_packed_bytes_per_layer(
        num_tokens=8,
        hidden_size=16,
        top_k=2,
        dtype_bytes=2,
    )
    assert bytes_per_layer == 1024


def test_compute_ep_grouped_bytes_per_layer() -> None:
    bytes_per_layer = benchmark_moe.compute_ep_grouped_bytes_per_layer(
        num_tokens=8,
        hidden_size=16,
        top_k=2,
        dtype_bytes=2,
    )
    assert bytes_per_layer == 1024


def test_build_comm_summary_contains_formulas() -> None:
    summary = benchmark_moe.build_comm_summary(
        num_tokens=32,
        hidden_size=64,
        top_k=2,
        dtype="float16",
        ep_size=2,
    )

    assert summary["tp_formula"] == "2 * num_tokens * hidden_size * dtype_bytes"
    assert summary["ep_ideal_formula"] == "2 * num_tokens * top_k * hidden_size * dtype_bytes"
    assert summary["ep_padded_formula"] == "2 * ep_size * num_tokens * top_k * hidden_size * dtype_bytes"
    assert summary["ep_packed_formula"] == "2 * num_tokens * top_k * hidden_size * dtype_bytes"
    assert summary["ep_grouped_formula"] == "2 * num_tokens * top_k * hidden_size * dtype_bytes"
    assert summary["ep_ideal_bytes_per_layer"] == summary["tp_bytes_per_layer"] * 2
    assert summary["ep_padded_bytes_per_layer"] == summary["ep_ideal_bytes_per_layer"] * 2
    assert summary["ep_packed_bytes_per_layer"] == summary["ep_ideal_bytes_per_layer"]
    assert summary["ep_grouped_bytes_per_layer"] == summary["ep_ideal_bytes_per_layer"]
    assert "comm_mode=padded" in summary["ep_padded_impl_note"]
    assert "comm_mode=packed" in summary["ep_packed_impl_note"]
    assert "control_plane_ms/control_plane_share" in summary["ep_packed_impl_note"]
    assert "expert_exec_mode=grouped" in summary["ep_grouped_impl_note"]
    assert "down_proj remains per-expert" in summary["ep_grouped_impl_note"]
    assert "resident local gate/up" in summary["ep_grouped_impl_note"]
    assert "runtime resident bytes separately" in summary["ep_grouped_impl_note"]
    assert "control_plane_ms/share" in summary["ep_packed_control_plane_note"]


def test_build_param_summary_reports_local_shard_ratio() -> None:
    args = benchmark_moe.build_argparser().parse_args(
        ["--num-experts", "4", "--ep-size", "2", "--src-rank", "1"]
    )
    layer = benchmark_moe.build_shared_layer(args)

    summary = benchmark_moe.build_param_summary(
        layer,
        ep_size=2,
        src_rank=1,
        runtime_dtype="float16",
    )

    assert summary["dense_param_bytes"] > summary["ep_rank_param_bytes"] > 0
    assert summary["expert_param_bytes"] > 0
    assert 0.5 < summary["shard_ratio"] < 1.0
    assert summary["dense_runtime_param_bytes"] > summary["ep_rank_runtime_param_bytes"] > 0
    assert summary["ep_grouped_runtime_gateup_cache_bytes"] > 0
    assert summary["ep_grouped_runtime_resident_bytes"] > summary["ep_rank_runtime_param_bytes"]
    assert summary["ep_grouped_runtime_resident_ratio"] > summary["shard_ratio"]


def test_build_param_summary_rejects_invalid_src_rank() -> None:
    args = benchmark_moe.build_argparser().parse_args(
        ["--num-experts", "4", "--ep-size", "2", "--src-rank", "2"]
    )
    layer = benchmark_moe.build_shared_layer(args)

    with pytest.raises(ValueError, match="src_rank"):
        benchmark_moe.build_param_summary(layer, ep_size=args.ep_size, src_rank=args.src_rank)


def test_build_hidden_states_shape_and_dtype() -> None:
    hidden_states = benchmark_moe.build_hidden_states(
        batch_size=2,
        seq_len=3,
        hidden_size=8,
        dtype=torch.float32,
        device="cpu",
        seed=7,
    )

    assert hidden_states.shape == (2, 3, 8)
    assert hidden_states.dtype == torch.float32


def test_build_argparser_defaults() -> None:
    parser = benchmark_moe.build_argparser()
    args = parser.parse_args([])

    assert args.mode == "dense"
    assert args.compare is False
    assert args.comm_mode == "padded"
    assert args.expert_exec_mode == "naive"
    assert args.ep_size == 2
    assert args.src_rank == 0
    assert args.top_k == 2
    assert args.hidden_size > 0


def test_resolve_dense_device_uses_src_rank_in_compare() -> None:
    parser = benchmark_moe.build_argparser()
    args = parser.parse_args(["--compare", "--device", "0", "--src-rank", "1"])

    dense_device = benchmark_moe.resolve_dense_device(args)

    assert dense_device == 1


def test_resolve_dense_device_rejects_invalid_src_rank_in_compare() -> None:
    parser = benchmark_moe.build_argparser()
    args = parser.parse_args(["--compare", "--ep-size", "2", "--src-rank", "2"])

    with pytest.raises(ValueError, match="src_rank"):
        benchmark_moe.resolve_dense_device(args)


def test_run_dry_run_does_not_instantiate_ep_engine(monkeypatch) -> None:
    class _FailingEngine:
        def __init__(self, *args, **kwargs):
            raise AssertionError("dry-run 不应实例化 EPEngine")

    monkeypatch.setattr(benchmark_moe, "EPEngine", _FailingEngine)
    args = benchmark_moe.build_argparser().parse_args(["--mode", "ep", "--dry-run"])

    result = benchmark_moe.run_dry_run(args)

    assert result["mode"] == "ep"
    assert result["comm"]["ep_padded_bytes_per_layer"] > 0
    assert result["comm"]["ep_packed_bytes_per_layer"] > 0
    assert result["comm"]["ep_grouped_bytes_per_layer"] == result["comm"]["ep_ideal_bytes_per_layer"]
    assert "expert_exec_mode=grouped" in result["comm"]["ep_grouped_impl_note"]
    assert result["params"]["dense_param_bytes"] > result["params"]["ep_rank_param_bytes"]
    assert result["params"]["expert_param_bytes"] > 0
    assert result["params"]["ep_grouped_runtime_gateup_cache_bytes"] > 0


def test_run_dry_run_rejects_invalid_src_rank() -> None:
    args = benchmark_moe.build_argparser().parse_args(
        ["--mode", "dense", "--dry-run", "--ep-size", "2", "--src-rank", "2"]
    )

    with pytest.raises(ValueError, match="src_rank"):
        benchmark_moe.run_dry_run(args)


def test_run_compare_benchmark_uses_shared_layer_and_inputs(monkeypatch) -> None:
    seen: dict[str, int] = {}
    seen_modes: list[tuple[str, str]] = []

    def fake_dense(args, shared_layer=None, shared_hidden_states=None, device=None):
        assert shared_layer is not None and shared_hidden_states is not None
        assert device == args.src_rank
        seen["dense_weight_ptr"] = shared_layer.router.gate.weight.data_ptr()
        seen["dense_input_ptr"] = shared_hidden_states.data_ptr()
        return {
            "mode": "dense",
            "throughput_tok_s": 10.0,
            "note": "dense",
            "output": torch.tensor([[1.0]]),
            "expert_loads": torch.tensor([1, 2]),
            "expert_score_sums": torch.tensor([0.4, 0.6]),
        }

    def fake_ep(
        args,
        shared_layer=None,
        shared_hidden_states=None,
        comm_mode=None,
        expert_exec_mode=None,
    ):
        assert shared_layer is not None and shared_hidden_states is not None
        assert comm_mode is not None
        selected_expert_exec_mode = "naive" if expert_exec_mode is None else expert_exec_mode
        seen[f"{comm_mode}_weight_ptr"] = shared_layer.router.gate.weight.data_ptr()
        seen[f"{comm_mode}_input_ptr"] = shared_hidden_states.data_ptr()
        seen_modes.append((comm_mode, selected_expert_exec_mode))
        if selected_expert_exec_mode == "grouped":
            mode = "ep_grouped"
            throughput = 22.0
            output = torch.tensor([[0.50]])
            control_plane_ms = 1.0
            control_plane_share = 0.05
        elif comm_mode == "padded":
            mode = "ep_padded"
            throughput = 20.0
            output = torch.tensor([[1.25]])
            control_plane_ms = 0.0
            control_plane_share = 0.0
        else:
            mode = "ep_packed"
            throughput = 18.0
            output = torch.tensor([[0.75]])
            control_plane_ms = 1.5
            control_plane_share = 0.1
        return {
            "mode": mode,
            "comm_mode": comm_mode,
            "expert_exec_mode": selected_expert_exec_mode,
            "throughput_tok_s": throughput,
            "note": f"{mode}_{selected_expert_exec_mode}",
            "control_plane_ms": control_plane_ms,
            "control_plane_share": control_plane_share,
            "control_plane_note": f"{mode}_cp",
            "output": output,
            "send_counts": torch.tensor([2, 2]),
            "expert_loads": torch.tensor([1, 2]),
            "expert_score_sums": torch.tensor([0.4, 0.6]),
            "elapsed_s": 0.1,
        }

    monkeypatch.setattr(benchmark_moe, "run_dense_benchmark", fake_dense)
    monkeypatch.setattr(benchmark_moe, "run_ep_benchmark", fake_ep)

    args = benchmark_moe.build_argparser().parse_args(["--compare"])
    result = benchmark_moe.run_compare_benchmark(args)

    assert seen_modes == [("padded", "naive"), ("packed", "naive"), ("packed", "grouped")]
    assert seen["dense_weight_ptr"] == seen["padded_weight_ptr"] == seen["packed_weight_ptr"]
    assert seen["dense_input_ptr"] == seen["padded_input_ptr"] == seen["packed_input_ptr"]
    assert result["max_abs_diff_padded"] == 0.25
    assert result["max_abs_diff_packed"] == 0.25
    assert result["max_abs_diff_grouped"] == 0.5
    assert result["ep_grouped"]["mode"] == "ep_grouped"


def test_run_ep_benchmark_reports_control_plane_metrics(monkeypatch) -> None:
    seen_kwargs: dict[str, object] = {}

    class _FakeEngine:
        def benchmark_forward(self, hidden_states, warmup, runs):
            assert hidden_states.shape == (2, 3, 8)
            assert warmup == 1
            assert runs == 2
            return {
                "elapsed_s": 0.5,
                "control_plane_elapsed_s": 0.05,
                "output": torch.tensor([[1.0]]),
                "send_counts": torch.tensor([2, 2]),
                "expert_loads": torch.tensor([2, 2]),
                "expert_score_sums": torch.tensor([0.4, 0.6]),
            }

    monkeypatch.setattr(
        benchmark_moe.EPEngine,
        "from_moe_layer",
        lambda *args, **kwargs: seen_kwargs.update(kwargs) or _FakeEngine(),
    )

    args = benchmark_moe.build_argparser().parse_args(
        [
            "--batch-size",
            "2",
            "--seq-len",
            "3",
            "--hidden-size",
            "8",
            "--warmup",
            "1",
            "--runs",
            "2",
            "--expert-exec-mode",
            "grouped",
        ]
    )
    result = benchmark_moe.run_ep_benchmark(args, comm_mode="packed", expert_exec_mode="grouped")

    assert seen_kwargs["comm_mode"] == "packed"
    assert seen_kwargs["expert_exec_mode"] == "grouped"
    assert result["mode"] == "ep_grouped"
    assert result["throughput_tok_s"] == 24.0
    assert result["control_plane_ms"] == 25.0
    assert result["control_plane_share"] == 0.1
    assert "PackedControlPlane" in result["control_plane_note"]
    assert "grouped local-expert count sync/helper" in result["control_plane_note"]
    assert result["expert_exec_mode"] == "grouped"
    assert "batched gate/up projections" in result["note"]
    assert "down_proj remains per-expert" in result["note"]
    assert "resident local gate/up packed-weight cache" in result["note"]


def test_run_dense_benchmark_times_only_gpu_work_on_selected_device(monkeypatch) -> None:
    events: list[tuple[str, object]] = []

    class _FakeHiddenStates:
        def to(self, device=None, dtype=None):
            events.append(("hidden_to", device))
            return self

    class _FakeCPUCopy:
        def __init__(self, name: str) -> None:
            self.name = name

        def cpu(self):
            events.append(("cpu", self.name))
            return self.name

    class _FakeLayer:
        def __call__(self, hidden_states, return_router_stats=False):
            phase = "timed" if return_router_stats else "warmup"
            events.append(("forward", phase))
            if not return_router_stats:
                return None
            return (
                _FakeCPUCopy("output"),
                None,
                SimpleNamespace(
                    expert_loads=_FakeCPUCopy("expert_loads"),
                    expert_score_sums=_FakeCPUCopy("expert_score_sums"),
                ),
            )

    perf_times = iter([100.0, 101.0])
    monkeypatch.setattr(benchmark_moe, "build_dense_layer", lambda *args, **kwargs: _FakeLayer())
    monkeypatch.setattr(benchmark_moe, "build_shared_hidden_states", lambda *args, **kwargs: _FakeHiddenStates())
    monkeypatch.setattr(benchmark_moe.time, "perf_counter", lambda: next(perf_times))
    monkeypatch.setattr(
        benchmark_moe.torch.cuda,
        "synchronize",
        lambda device=None: events.append(("sync", device)),
    )

    args = benchmark_moe.build_argparser().parse_args(
        ["--batch-size", "2", "--seq-len", "3", "--dtype", "float32", "--warmup", "1", "--runs", "1"]
    )
    result = benchmark_moe.run_dense_benchmark(args, device=1)

    assert result["throughput_tok_s"] == 6.0
    assert result["output"] == "output"
    assert result["expert_loads"] == "expert_loads"
    assert result["expert_score_sums"] == "expert_score_sums"
    assert events == [
        ("hidden_to", "cuda:1"),
        ("forward", "warmup"),
        ("sync", 1),
        ("forward", "timed"),
        ("sync", 1),
        ("cpu", "expert_loads"),
        ("cpu", "expert_score_sums"),
        ("cpu", "output"),
    ]
