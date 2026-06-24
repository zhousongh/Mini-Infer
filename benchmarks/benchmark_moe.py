"""Phase 17-21：synthetic MoE / Expert Parallel benchmark。

对比对象：
- dense `MoELayer`
- `EPEngine`（2 卡，all-to-all + expert dispatch/gather）

输出内容：
- dense / EP token throughput
- per-rank send_counts 与 per-expert token load
- TP vs EP 通信量公式

当前口径：
- 输入是 synthetic hidden states，不接真实 HuggingFace 权重
- `--mode ep` 的正式计时在 worker 内部完成，默认排除 `mp.spawn` 与进程组初始化开销
- `--compare` 使用同一组权重和同一组 hidden states 对比 dense / `ep_padded` / `ep_packed` / `ep_grouped`
- 参数量结果同时区分：
  - dense 全量参数
  - EP 单 rank local shard 参数
- 通信结果同时区分：
  - ideal EP hidden-state bytes（按真实 token 副本数估算）
  - current padded prototype bytes（按当前 `all_to_all_single` fixed chunk 实现估算）
  - Phase 19 packed bytes（按 exact split-size hidden-state payload 估算）
- Phase 20 继续显式输出 packed control-plane 指标（`control_plane_ms` / `control_plane_share`）
- Phase 21 继续增加 grouped local expert execution 对照（`expert_exec_mode=grouped`）
- `--dry-run` 只验证参数构造、通信量公式和 benchmark 主流程
"""

from __future__ import annotations

import argparse
import time

import torch

from mini_infer.modeling.moe_layer import MoELayer, shard_moe_state_dict
from mini_infer.modeling.moe_model import SyntheticMoEConfig
from mini_infer.parallel.ep_engine import EPEngine

_DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def validate_src_rank(src_rank: int, ep_size: int) -> None:
    """统一校验 benchmark 入口的 source rank。"""
    if src_rank < 0 or src_rank >= ep_size:
        raise ValueError(
            f"src_rank 必须落在 [0, ep_size) 内，当前 src_rank={src_rank}, ep_size={ep_size}"
        )


def build_hidden_states(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: str,
    seed: int = 0,
) -> torch.Tensor:
    """构造可复现的 synthetic hidden states。"""
    torch.manual_seed(seed)
    return torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)


def build_shared_layer(args: argparse.Namespace) -> MoELayer:
    """构造 compare/dense/ep 共用的 base layer。"""
    torch.manual_seed(args.seed)
    return MoELayer(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        bias=args.bias,
    ).float().eval()


def build_shared_hidden_states(args: argparse.Namespace) -> torch.Tensor:
    """构造 compare/dense/ep 共用的 CPU hidden states。"""
    return build_hidden_states(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        hidden_size=args.hidden_size,
        dtype=torch.float32,
        device="cpu",
        seed=args.seed + 1,
    )


def resolve_dense_device(args: argparse.Namespace) -> int:
    """compare 模式下 dense 与 EP 共享同一 source device 口径。"""
    if args.compare:
        validate_src_rank(args.src_rank, args.ep_size)
        return args.src_rank
    return args.device


def compute_tp_bytes_per_layer(
    num_tokens: int,
    hidden_size: int,
    dtype_bytes: int,
) -> int:
    """粗略估算 dense TP 在一层上的通信量。"""
    return 2 * num_tokens * hidden_size * dtype_bytes


def compute_ep_bytes_per_layer(
    num_tokens: int,
    hidden_size: int,
    top_k: int,
    dtype_bytes: int,
) -> int:
    """估算 ideal EP 的 dispatch + gather hidden-state 通信量。"""
    return 2 * num_tokens * top_k * hidden_size * dtype_bytes


def compute_ep_padded_bytes_per_layer(
    num_tokens: int,
    hidden_size: int,
    top_k: int,
    dtype_bytes: int,
    ep_size: int,
) -> int:
    """估算当前 padded EP prototype 的 hidden-state 通信量。"""
    return 2 * ep_size * num_tokens * top_k * hidden_size * dtype_bytes


def compute_ep_packed_bytes_per_layer(
    num_tokens: int,
    hidden_size: int,
    top_k: int,
    dtype_bytes: int,
) -> int:
    """估算 Phase 19 packed EP 的 hidden-state 通信量。"""
    return 2 * num_tokens * top_k * hidden_size * dtype_bytes


def compute_ep_grouped_bytes_per_layer(
    num_tokens: int,
    hidden_size: int,
    top_k: int,
    dtype_bytes: int,
) -> int:
    """Phase 21 grouped local expert execution 沿用 packed exact payload bytes。"""
    return 2 * num_tokens * top_k * hidden_size * dtype_bytes


def build_comm_summary(
    num_tokens: int,
    hidden_size: int,
    top_k: int,
    dtype: str,
    ep_size: int,
) -> dict[str, object]:
    dtype_bytes = torch.tensor([], dtype=_DTYPE_MAP[dtype]).element_size()
    tp_bytes = compute_tp_bytes_per_layer(num_tokens, hidden_size, dtype_bytes)
    ep_ideal_bytes = compute_ep_bytes_per_layer(num_tokens, hidden_size, top_k, dtype_bytes)
    ep_padded_bytes = compute_ep_padded_bytes_per_layer(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        top_k=top_k,
        dtype_bytes=dtype_bytes,
        ep_size=ep_size,
    )
    ep_packed_bytes = compute_ep_packed_bytes_per_layer(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        top_k=top_k,
        dtype_bytes=dtype_bytes,
    )
    ep_grouped_bytes = compute_ep_grouped_bytes_per_layer(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        top_k=top_k,
        dtype_bytes=dtype_bytes,
    )
    return {
        "dtype": dtype,
        "dtype_bytes": dtype_bytes,
        "tp_bytes_per_layer": tp_bytes,
        "ep_ideal_bytes_per_layer": ep_ideal_bytes,
        "ep_padded_bytes_per_layer": ep_padded_bytes,
        "ep_packed_bytes_per_layer": ep_packed_bytes,
        "ep_grouped_bytes_per_layer": ep_grouped_bytes,
        "tp_formula": "2 * num_tokens * hidden_size * dtype_bytes",
        "ep_ideal_formula": "2 * num_tokens * top_k * hidden_size * dtype_bytes",
        "ep_padded_formula": "2 * ep_size * num_tokens * top_k * hidden_size * dtype_bytes",
        "ep_packed_formula": "2 * num_tokens * top_k * hidden_size * dtype_bytes",
        "ep_grouped_formula": "2 * num_tokens * top_k * hidden_size * dtype_bytes",
        "ep_padded_impl_note": (
            "current EPMoELayer comm_mode=padded uses fixed-size all_to_all_single chunks; "
            "hidden-state bytes only, expert-id/valid metadata excluded"
        ),
        "ep_packed_impl_note": (
            "Phase 20 comm_mode=packed uses exact split-size all_to_all_single; "
            "hidden-state bytes only, expert-id/control-plane metadata excluded; "
            "benchmark additionally reports control_plane_ms/control_plane_share"
        ),
        "ep_grouped_impl_note": (
            "Phase 21 expert_exec_mode=grouped keeps packed exact split-size all_to_all_single; "
            "local expert execution switches from per-expert where/index_select/index_copy_ "
            "to grouped contiguous slices + batched gate/up projections while down_proj remains per-expert; "
            "hidden-state bytes remain identical to packed/ideal; grouped mode keeps a resident local gate/up "
            "packed-weight cache and benchmark reports the extra runtime resident bytes separately"
        ),
        "ep_packed_control_plane_note": (
            "control_plane_ms/share measure worker-side packed split-size control plane only; "
            "source-rank router/dispatch GPU work remains included in packed throughput"
        ),
    }


def compute_state_dict_bytes(
    state_dict: dict[str, torch.Tensor],
    prefix: str | None = None,
) -> int:
    """统计 state_dict 中张量占用的总字节数。"""
    total = 0
    for key, value in state_dict.items():
        if prefix is not None and not key.startswith(prefix):
            continue
        total += value.numel() * value.element_size()
    return total


def compute_state_dict_numel(
    state_dict: dict[str, torch.Tensor],
    prefix: str | None = None,
) -> int:
    total = 0
    for key, value in state_dict.items():
        if prefix is not None and not key.startswith(prefix):
            continue
        total += value.numel()
    return total


def build_param_summary(
    layer: MoELayer,
    ep_size: int,
    src_rank: int,
    runtime_dtype: str | None = None,
) -> dict[str, object]:
    """统计 dense 参数量与单 rank local shard 参数量。"""
    validate_src_rank(src_rank, ep_size)
    dense_state_dict = layer.state_dict()
    rank_state_dicts = shard_moe_state_dict(
        dense_state_dict,
        num_experts=layer.num_experts,
        ep_size=ep_size,
    )
    dense_param_bytes = compute_state_dict_bytes(dense_state_dict)
    expert_param_bytes = compute_state_dict_bytes(dense_state_dict, prefix="experts.")
    ep_rank_param_bytes = compute_state_dict_bytes(rank_state_dicts[src_rank])
    shard_ratio = ep_rank_param_bytes / dense_param_bytes
    summary = {
        "dense_param_bytes": dense_param_bytes,
        "ep_rank_param_bytes": ep_rank_param_bytes,
        "expert_param_bytes": expert_param_bytes,
        "shard_ratio": shard_ratio,
    }
    if runtime_dtype is not None:
        runtime_dtype_bytes = torch.empty((), dtype=_DTYPE_MAP[runtime_dtype]).element_size()
        dense_param_numel = compute_state_dict_numel(dense_state_dict)
        ep_rank_param_numel = compute_state_dict_numel(rank_state_dicts[src_rank])
        local_experts = layer.num_experts // ep_size
        grouped_gateup_cache_numel = 2 * local_experts * layer.hidden_size * layer.intermediate_size
        if layer.experts[0].gate_proj.bias is not None:
            grouped_gateup_cache_numel += 2 * local_experts * layer.intermediate_size
        dense_runtime_param_bytes = dense_param_numel * runtime_dtype_bytes
        ep_rank_runtime_param_bytes = ep_rank_param_numel * runtime_dtype_bytes
        ep_grouped_runtime_gateup_cache_bytes = grouped_gateup_cache_numel * runtime_dtype_bytes
        ep_grouped_runtime_resident_bytes = ep_rank_runtime_param_bytes + ep_grouped_runtime_gateup_cache_bytes
        summary.update(
            {
                "dense_runtime_param_bytes": dense_runtime_param_bytes,
                "ep_rank_runtime_param_bytes": ep_rank_runtime_param_bytes,
                "ep_grouped_runtime_gateup_cache_bytes": ep_grouped_runtime_gateup_cache_bytes,
                "ep_grouped_runtime_resident_bytes": ep_grouped_runtime_resident_bytes,
                "ep_grouped_runtime_resident_ratio": ep_grouped_runtime_resident_bytes / dense_runtime_param_bytes,
                "ep_grouped_runtime_resident_note": (
                    "grouped mode keeps a resident local gate/up packed-weight cache; "
                    "runtime resident bytes include this cache in addition to the rank-local shard"
                ),
            }
        )
    return summary


def build_dense_layer(
    args: argparse.Namespace,
    shared_layer: MoELayer | None = None,
    device: int | None = None,
) -> MoELayer:
    layer = MoELayer(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        bias=args.bias,
    )
    if shared_layer is not None:
        layer.load_state_dict(shared_layer.state_dict(), strict=True)
    dense_device = args.device if device is None else device
    return layer.to(device=f"cuda:{dense_device}", dtype=_DTYPE_MAP[args.dtype]).eval()


def run_dense_benchmark(
    args: argparse.Namespace,
    shared_layer: MoELayer | None = None,
    shared_hidden_states: torch.Tensor | None = None,
    device: int | None = None,
) -> dict[str, object]:
    dense_device = args.device if device is None else device
    layer = build_dense_layer(args, shared_layer=shared_layer, device=dense_device)
    hidden_states_cpu = shared_hidden_states if shared_hidden_states is not None else build_shared_hidden_states(args)
    hidden_states = hidden_states_cpu.to(device=f"cuda:{dense_device}", dtype=_DTYPE_MAP[args.dtype])

    aux = None
    output_cpu = None
    with torch.no_grad():
        for _ in range(args.warmup):
            layer(hidden_states)
        torch.cuda.synchronize(device=dense_device)
        t0 = time.perf_counter()
        output = None
        stats = None
        for _ in range(args.runs):
            output, _, stats = layer(hidden_states, return_router_stats=True)
        torch.cuda.synchronize(device=dense_device)
        elapsed = time.perf_counter() - t0
        if output is not None and stats is not None:
            aux = {
                "expert_loads": stats.expert_loads.cpu(),
                "expert_score_sums": stats.expert_score_sums.cpu(),
            }
            output_cpu = output.cpu()

    num_tokens = args.batch_size * args.seq_len * args.runs
    throughput = num_tokens / elapsed
    result = {
        "mode": "dense",
        "throughput_tok_s": throughput,
        "note": f"single GPU dense MoELayer on cuda:{dense_device}",
        "output": output_cpu,
    }
    if aux is not None:
        result.update(aux)
    return result


def run_ep_benchmark(
    args: argparse.Namespace,
    shared_layer: MoELayer | None = None,
    shared_hidden_states: torch.Tensor | None = None,
    comm_mode: str | None = None,
    expert_exec_mode: str | None = None,
) -> dict[str, object]:
    dense_layer = shared_layer if shared_layer is not None else build_shared_layer(args)
    selected_comm_mode = args.comm_mode if comm_mode is None else comm_mode
    selected_expert_exec_mode = args.expert_exec_mode if expert_exec_mode is None else expert_exec_mode
    engine = EPEngine.from_moe_layer(
        dense_layer,
        ep_size=args.ep_size,
        dtype=args.dtype,
        src_rank=args.src_rank,
        comm_mode=selected_comm_mode,
        expert_exec_mode=selected_expert_exec_mode,
    )
    hidden_states = shared_hidden_states if shared_hidden_states is not None else build_shared_hidden_states(args)

    bench = engine.benchmark_forward(
        hidden_states,
        warmup=args.warmup,
        runs=args.runs,
    )
    elapsed = float(bench["elapsed_s"])
    control_plane_elapsed_s = float(bench.get("control_plane_elapsed_s", 0.0) or 0.0)

    num_tokens = args.batch_size * args.seq_len * args.runs
    throughput = num_tokens / elapsed
    if selected_expert_exec_mode == "grouped":
        mode = "ep_grouped"
    else:
        mode = f"ep_{selected_comm_mode}"
    note = (
        "2-GPU EPEngine steady-state worker timing; spawn/init excluded; "
        f"comm_mode={selected_comm_mode}; expert_exec_mode={selected_expert_exec_mode}"
    )
    if selected_comm_mode == "packed":
        note += "; includes source-rank router/dispatch + split-size control plane"
    if selected_expert_exec_mode == "grouped":
        note += "; local expert execution uses grouped contiguous slices + batched gate/up projections while down_proj remains per-expert; grouped mode keeps a resident local gate/up packed-weight cache"
    if selected_comm_mode == "packed":
        if selected_expert_exec_mode == "grouped":
            control_plane_note = (
                "per-run packed control plane = GPU send-count sync to host + PackedControlPlane split-size helper "
                "+ grouped local-expert count sync/helper; source-rank router/dispatch GPU work excluded from this metric"
            )
        else:
            control_plane_note = (
                "per-run packed control plane = GPU send-count sync to host + PackedControlPlane split-size helper; "
                "source-rank router/dispatch GPU work excluded from this metric"
            )
    else:
        control_plane_note = "comm_mode=padded has no packed split-size control plane"
    result = {
        "mode": mode,
        "comm_mode": selected_comm_mode,
        "expert_exec_mode": selected_expert_exec_mode,
        "throughput_tok_s": throughput,
        "note": note,
        "output": bench["output"],
        "send_counts": bench["send_counts"],
        "expert_loads": bench["expert_loads"],
        "expert_score_sums": bench["expert_score_sums"],
        "elapsed_s": elapsed,
        "control_plane_ms": 1000.0 * control_plane_elapsed_s / args.runs if args.runs > 0 else 0.0,
        "control_plane_share": control_plane_elapsed_s / elapsed if elapsed > 0 else 0.0,
        "control_plane_note": control_plane_note,
    }
    return result


def run_compare_benchmark(args: argparse.Namespace) -> dict[str, object]:
    """用同一组权重和 hidden states 对比 dense / ep_padded / ep_packed / ep_grouped。"""
    shared_layer = build_shared_layer(args)
    shared_hidden_states = build_shared_hidden_states(args)
    dense_device = resolve_dense_device(args)
    dense_result = run_dense_benchmark(
        args,
        shared_layer=shared_layer,
        shared_hidden_states=shared_hidden_states,
        device=dense_device,
    )
    ep_padded_result = run_ep_benchmark(
        args,
        shared_layer=shared_layer,
        shared_hidden_states=shared_hidden_states,
        comm_mode="padded",
    )
    ep_packed_result = run_ep_benchmark(
        args,
        shared_layer=shared_layer,
        shared_hidden_states=shared_hidden_states,
        comm_mode="packed",
    )
    ep_grouped_result = run_ep_benchmark(
        args,
        shared_layer=shared_layer,
        shared_hidden_states=shared_hidden_states,
        comm_mode="packed",
        expert_exec_mode="grouped",
    )
    assert dense_result["output"] is not None
    assert ep_padded_result["output"] is not None
    assert ep_packed_result["output"] is not None
    assert ep_grouped_result["output"] is not None
    max_abs_diff_padded = (
        dense_result["output"].float() - ep_padded_result["output"].float()
    ).abs().max().item()
    max_abs_diff_packed = (
        dense_result["output"].float() - ep_packed_result["output"].float()
    ).abs().max().item()
    max_abs_diff_grouped = (
        dense_result["output"].float() - ep_grouped_result["output"].float()
    ).abs().max().item()
    return {
        "dense": dense_result,
        "ep_padded": ep_padded_result,
        "ep_packed": ep_packed_result,
        "ep_grouped": ep_grouped_result,
        "max_abs_diff_padded": float(max_abs_diff_padded),
        "max_abs_diff_packed": float(max_abs_diff_packed),
        "max_abs_diff_grouped": float(max_abs_diff_grouped),
    }


def run_dry_run(args: argparse.Namespace) -> dict[str, object]:
    """只做参数和通信量口径验证，不依赖 GPU。"""
    cfg = SyntheticMoEConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )
    shared_layer = build_shared_layer(args)
    _ = build_shared_hidden_states(args)
    comm = build_comm_summary(
        num_tokens=args.batch_size * args.seq_len,
        hidden_size=cfg.hidden_size,
        top_k=cfg.top_k,
        dtype=args.dtype,
        ep_size=args.ep_size,
    )
    params = build_param_summary(
        shared_layer,
        ep_size=args.ep_size,
        src_rank=args.src_rank,
        runtime_dtype=args.dtype,
    )
    return {
        "mode": "compare" if args.compare else args.mode,
        "comm": comm,
        "params": params,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 21 grouped expert execution benchmark")
    parser.add_argument("--mode", choices=["dense", "ep"], default="dense")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--intermediate-size", type=int, default=1024)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--ep-size", type=int, default=2)
    parser.add_argument("--dtype", choices=sorted(_DTYPE_MAP), default="float16")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--src-rank", type=int, default=0)
    parser.add_argument("--comm-mode", choices=["padded", "packed"], default="padded")
    parser.add_argument("--expert-exec-mode", choices=["naive", "grouped"], default="naive")
    parser.add_argument("--bias", action="store_true")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.dry_run:
        dry_run = run_dry_run(args)
        comm = dry_run["comm"]
        params = dry_run["params"]
        print("=== MoE / EP benchmark dry-run ===")
        print(f"mode={dry_run['mode']}")
        print(
            f"batch_size={args.batch_size}, seq_len={args.seq_len}, hidden={args.hidden_size}, "
            f"intermediate={args.intermediate_size}, num_experts={args.num_experts}, top_k={args.top_k}"
        )
        print(f"tp_formula={comm['tp_formula']}")
        print(f"ep_ideal_formula={comm['ep_ideal_formula']}")
        print(f"ep_padded_formula={comm['ep_padded_formula']}")
        print(f"ep_packed_formula={comm['ep_packed_formula']}")
        print(f"ep_grouped_formula={comm['ep_grouped_formula']}")
        print(f"tp_bytes_per_layer={comm['tp_bytes_per_layer']}")
        print(f"ep_ideal_bytes_per_layer={comm['ep_ideal_bytes_per_layer']}")
        print(f"ep_padded_bytes_per_layer={comm['ep_padded_bytes_per_layer']}")
        print(f"ep_packed_bytes_per_layer={comm['ep_packed_bytes_per_layer']}")
        print(f"ep_grouped_bytes_per_layer={comm['ep_grouped_bytes_per_layer']}")
        print(f"dense_param_bytes={params['dense_param_bytes']}")
        print(f"ep_rank_param_bytes={params['ep_rank_param_bytes']}")
        print(f"expert_param_bytes={params['expert_param_bytes']}")
        print(f"shard_ratio={params['shard_ratio']:.4f}")
        print(f"dense_runtime_param_bytes={params['dense_runtime_param_bytes']}")
        print(f"ep_rank_runtime_param_bytes={params['ep_rank_runtime_param_bytes']}")
        print(f"ep_grouped_runtime_gateup_cache_bytes={params['ep_grouped_runtime_gateup_cache_bytes']}")
        print(f"ep_grouped_runtime_resident_bytes={params['ep_grouped_runtime_resident_bytes']}")
        print(f"ep_grouped_runtime_resident_ratio={params['ep_grouped_runtime_resident_ratio']:.4f}")
        print(f"ep_grouped_runtime_resident_note={params['ep_grouped_runtime_resident_note']}")
        print(f"ep_padded_impl_note={comm['ep_padded_impl_note']}")
        print(f"ep_packed_impl_note={comm['ep_packed_impl_note']}")
        print(f"ep_grouped_impl_note={comm['ep_grouped_impl_note']}")
        print(f"ep_packed_control_plane_note={comm['ep_packed_control_plane_note']}")
        print("dry_run=ok")
        return

    param_summary = build_param_summary(
        build_shared_layer(args),
        ep_size=args.ep_size,
        src_rank=args.src_rank,
        runtime_dtype=args.dtype,
    )
    comm = build_comm_summary(
        num_tokens=args.batch_size * args.seq_len,
        hidden_size=args.hidden_size,
        top_k=args.top_k,
        dtype=args.dtype,
        ep_size=args.ep_size,
    )
    if args.compare:
        result = run_compare_benchmark(args)
        dense = result["dense"]
        ep_padded = result["ep_padded"]
        ep_packed = result["ep_packed"]
        ep_grouped = result["ep_grouped"]
        print("=== MoE / EP benchmark (compare) ===")
        print(f"dense_throughput_tok_s={dense['throughput_tok_s']:.2f}")
        print(f"ep_padded_throughput_tok_s={ep_padded['throughput_tok_s']:.2f}")
        print(f"ep_packed_throughput_tok_s={ep_packed['throughput_tok_s']:.2f}")
        print(f"ep_grouped_throughput_tok_s={ep_grouped['throughput_tok_s']:.2f}")
        print(f"max_abs_diff_padded={result['max_abs_diff_padded']:.6f}")
        print(f"max_abs_diff_packed={result['max_abs_diff_packed']:.6f}")
        print(f"max_abs_diff_grouped={result['max_abs_diff_grouped']:.6f}")
        print(f"dense_note={dense['note']}")
        print(f"ep_padded_note={ep_padded['note']}")
        print(f"ep_packed_note={ep_packed['note']}")
        print(f"ep_grouped_note={ep_grouped['note']}")
        print(f"ep_padded_control_plane_ms={ep_padded['control_plane_ms']:.4f}")
        print(f"ep_padded_control_plane_share={ep_padded['control_plane_share']:.6f}")
        print(f"ep_packed_control_plane_ms={ep_packed['control_plane_ms']:.4f}")
        print(f"ep_packed_control_plane_share={ep_packed['control_plane_share']:.6f}")
        print(f"ep_grouped_control_plane_ms={ep_grouped['control_plane_ms']:.4f}")
        print(f"ep_grouped_control_plane_share={ep_grouped['control_plane_share']:.6f}")
        print(f"ep_packed_control_plane_note={ep_packed['control_plane_note']}")
        print(f"ep_grouped_control_plane_note={ep_grouped['control_plane_note']}")
        print(f"ep_padded_send_counts={ep_padded['send_counts'].tolist()}")
        print(f"ep_packed_send_counts={ep_packed['send_counts'].tolist()}")
        print(f"ep_grouped_send_counts={ep_grouped['send_counts'].tolist()}")
        print(f"dense_expert_loads={dense['expert_loads'].tolist()}")
        print(f"ep_padded_expert_loads={ep_padded['expert_loads'].tolist()}")
        print(f"ep_packed_expert_loads={ep_packed['expert_loads'].tolist()}")
        print(f"ep_grouped_expert_loads={ep_grouped['expert_loads'].tolist()}")
        print(
            f"ep_padded_expert_score_sums={[round(float(v), 4) for v in ep_padded['expert_score_sums']]}"
        )
        print(
            f"ep_packed_expert_score_sums={[round(float(v), 4) for v in ep_packed['expert_score_sums']]}"
        )
        print(
            f"ep_grouped_expert_score_sums={[round(float(v), 4) for v in ep_grouped['expert_score_sums']]}"
        )
    else:
        if args.mode == "dense":
            result = run_dense_benchmark(args)
        else:
            result = run_ep_benchmark(
                args,
                comm_mode=args.comm_mode,
                expert_exec_mode=args.expert_exec_mode,
            )

        print(f"=== MoE / EP benchmark ({result['mode']}) ===")
        print(f"throughput_tok_s={result['throughput_tok_s']:.2f}")
        print(f"note={result['note']}")
        if "control_plane_ms" in result:
            print(f"control_plane_ms={result['control_plane_ms']:.4f}")
        if "control_plane_share" in result:
            print(f"control_plane_share={result['control_plane_share']:.6f}")
        if "control_plane_note" in result:
            print(f"control_plane_note={result['control_plane_note']}")
        if "send_counts" in result:
            print(f"send_counts={result['send_counts'].tolist()}")
        if "expert_loads" in result:
            print(f"expert_loads={result['expert_loads'].tolist()}")
        if "expert_score_sums" in result:
            print(f"expert_score_sums={[round(float(v), 4) for v in result['expert_score_sums']]}")
    print(f"tp_formula={comm['tp_formula']}")
    print(f"ep_ideal_formula={comm['ep_ideal_formula']}")
    print(f"ep_padded_formula={comm['ep_padded_formula']}")
    print(f"ep_packed_formula={comm['ep_packed_formula']}")
    print(f"ep_grouped_formula={comm['ep_grouped_formula']}")
    print(f"tp_bytes_per_layer={comm['tp_bytes_per_layer']}")
    print(f"ep_ideal_bytes_per_layer={comm['ep_ideal_bytes_per_layer']}")
    print(f"ep_padded_bytes_per_layer={comm['ep_padded_bytes_per_layer']}")
    print(f"ep_packed_bytes_per_layer={comm['ep_packed_bytes_per_layer']}")
    print(f"ep_grouped_bytes_per_layer={comm['ep_grouped_bytes_per_layer']}")
    print(f"dense_param_bytes={param_summary['dense_param_bytes']}")
    print(f"ep_rank_param_bytes={param_summary['ep_rank_param_bytes']}")
    print(f"expert_param_bytes={param_summary['expert_param_bytes']}")
    print(f"shard_ratio={param_summary['shard_ratio']:.4f}")
    print(f"dense_runtime_param_bytes={param_summary['dense_runtime_param_bytes']}")
    print(f"ep_rank_runtime_param_bytes={param_summary['ep_rank_runtime_param_bytes']}")
    print(f"ep_grouped_runtime_gateup_cache_bytes={param_summary['ep_grouped_runtime_gateup_cache_bytes']}")
    print(f"ep_grouped_runtime_resident_bytes={param_summary['ep_grouped_runtime_resident_bytes']}")
    print(f"ep_grouped_runtime_resident_ratio={param_summary['ep_grouped_runtime_resident_ratio']:.4f}")
    print(f"ep_grouped_runtime_resident_note={param_summary['ep_grouped_runtime_resident_note']}")
    print(f"ep_padded_impl_note={comm['ep_padded_impl_note']}")
    print(f"ep_packed_impl_note={comm['ep_packed_impl_note']}")
    print(f"ep_grouped_impl_note={comm['ep_grouped_impl_note']}")
    print(f"ep_packed_control_plane_note={comm['ep_packed_control_plane_note']}")


if __name__ == "__main__":
    main()
