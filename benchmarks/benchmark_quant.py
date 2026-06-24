"""Phase 16：W8A8 / mixed-fallback 量化 benchmark。

对比 fp16 vs W8A8 的：
- 权重显存占用（MB）
- prefill / decode / e2e 三段耗时
- decode 吞吐（tokens/s）与 TPOT（ms/token）
- e2e 吞吐（tokens/s）
- greedy token match rate / sequence exact match
- W8A8 quant 路径中 _int_mm vs fallback 命中情况

评估口径：
- 固定使用 12 条 prompt 做正确性对比
- `batch_size` 表示引擎每次处理的 engine batch 大小，而不是评估 prompt 总数
- 模型结构参数从 HuggingFace config 推导，不再把 benchmark 限死在 1.5B
- `mode=w8a8` 表示启用 int8 权重存储路径；若 workload 中 fallback 占主导，
  benchmark 结果代表 mixed compute path，而不是纯 A8 matmul

用法示例：
  # 单模式运行
  HF_HUB_OFFLINE=1 conda run -n ai-infra python benchmarks/benchmark_quant.py \\
      --model <path> --mode fp16 --batch-size 4 --max-new-tokens 32

  # 对比运行（fp16 + W8A8 一次性比较）
  HF_HUB_OFFLINE=1 conda run -n ai-infra python benchmarks/benchmark_quant.py \\
      --model <path> --compare --batch-size 4 --max-new-tokens 64

  # 对当前量化 contract 做线性层 M-sweep，观察 _int_mm 何时命中
  HF_HUB_OFFLINE=1 conda run -n ai-infra python benchmarks/benchmark_quant.py \\
      --model <path> --compare --linear-bench-iters 50
"""

import argparse
import gc
import time
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig

from mini_infer.cache.kv_cache import KVCacheManager
from mini_infer.core.config import EngineConfig
from mini_infer.core.request import Request, RequestState, SamplingParams
from mini_infer.modeling.model_runner import ModelRunner
from mini_infer.modeling.quantization import QuantLinear

# --------------------------------------------------------------------------- #
# 固定 prompt 集（exact match 评估用，12 条）
# --------------------------------------------------------------------------- #

PROMPTS = [
    "The capital of France is",
    "Python is a programming language that",
    "The largest planet in the solar system is",
    "Water boils at",
    "The speed of light is approximately",
    "Albert Einstein was born in",
    "The chemical symbol for gold is",
    "The first element in the periodic table is",
    "Mount Everest is located in",
    "Shakespeare wrote",
    "The Great Wall of China was built during",
    "DNA stands for",
]

LINEAR_SWEEP_ROWS = (1, 2, 4, 8, 16, 32, 64)


def measure_weight_memory(model: torch.nn.Module) -> float:
    """统计模型所有参数和 buffer 的 GPU 显存（MB）。"""
    total_bytes = 0
    for t in list(model.parameters()) + list(model.buffers()):
        if t.is_cuda:
            total_bytes += t.numel() * t.element_size()
    return total_bytes / 1024 / 1024


def compute_match_metrics(
    reference_token_ids: list[list[int]],
    candidate_token_ids: list[list[int]],
) -> dict[str, float]:
    """计算 greedy token match rate 和 sequence exact match。

    口径：
      - token_match_rate：逐位置对齐比较；长度差异按 mismatch 计入分母
      - sequence_exact_rate：整段 token 序列完全相同的 prompt 占比
    """
    if len(reference_token_ids) != len(candidate_token_ids):
        raise ValueError("reference_token_ids 与 candidate_token_ids 长度不一致")

    token_matches = 0
    token_total = 0
    sequence_exact = 0

    for ref_ids, cand_ids in zip(reference_token_ids, candidate_token_ids):
        if ref_ids == cand_ids:
            sequence_exact += 1
        token_total += max(len(ref_ids), len(cand_ids))
        token_matches += sum(
            1
            for idx in range(max(len(ref_ids), len(cand_ids)))
            if idx < len(ref_ids)
            and idx < len(cand_ids)
            and ref_ids[idx] == cand_ids[idx]
        )

    prompt_total = len(reference_token_ids)
    return {
        "token_match_rate": (token_matches / token_total) if token_total > 0 else 1.0,
        "token_matches": float(token_matches),
        "token_total": float(token_total),
        "sequence_exact_rate": (sequence_exact / prompt_total) if prompt_total > 0 else 1.0,
        "sequence_exact_count": float(sequence_exact),
        "prompt_total": float(prompt_total),
    }


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _accumulate_quant_stats(total: Optional[dict[str, int]], current: Optional[dict[str, int]]) -> Optional[dict[str, int]]:
    if current is None:
        return total
    if total is None:
        return dict(current)
    for key, value in current.items():
        total[key] = total.get(key, 0) + int(value)
    return total


def _format_quant_stats(stats: Optional[dict[str, int]]) -> str:
    if not stats:
        return "N/A"
    total_rows = stats["int_mm_rows"] + stats["fallback_rows"]
    fallback_ratio = (stats["fallback_rows"] / total_rows) if total_rows > 0 else 0.0
    return (
        f"int_mm_calls={stats['int_mm_calls']}, fallback_calls={stats['fallback_calls']}, "
        f"fallback_rows_ratio={fallback_ratio:.1%}"
    )


def format_linear_quant_stats(stats: Optional[dict[str, int]]) -> str:
    """格式化 linear sweep 中单个 M 的量化路径统计。"""
    return _format_quant_stats(stats)


def _fallback_rows_ratio(stats: Optional[dict[str, int]]) -> Optional[float]:
    if not stats:
        return None
    total_rows = stats["int_mm_rows"] + stats["fallback_rows"]
    if total_rows <= 0:
        return 0.0
    return stats["fallback_rows"] / total_rows


def build_quant_compute_note(
    contract: Optional[dict[str, object]],
    decode_quant_stats: Optional[dict[str, int]],
) -> Optional[str]:
    """生成当前 quant compute 路径说明，避免把 mixed fallback 误写成纯 W8A8。"""
    if not contract:
        return None

    note = (
        f"weight_storage={contract['weight_storage']}; "
        f"int_mm_activation={contract['int_mm_activation_granularity']} (M>={contract['int_mm_min_rows']}); "
        f"fallback_activation={contract['fallback_activation_granularity']}; "
        f"fallback_compute={contract['fallback_compute']}"
    )

    fallback_ratio = _fallback_rows_ratio(decode_quant_stats)
    if fallback_ratio is not None and fallback_ratio > 0.0:
        note += "; fallback dequantizes weight per forward"
        if fallback_ratio >= 0.999:
            note += "; current decode result is mixed fallback compute, not pure A8 matmul"
    return note


def build_prompt_batches(
    batch_size: int,
    prompts: Optional[list[str]] = None,
) -> list[list[str]]:
    """把固定 prompt 集切成多个 engine batch。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    prompt_list = PROMPTS if prompts is None else prompts
    return [
        prompt_list[start:start + batch_size]
        for start in range(0, len(prompt_list), batch_size)
    ]


def build_linear_sweep_rows(max_rows: int = 64) -> list[int]:
    """构造 <= max_rows 的标准 M-sweep。"""
    return [rows for rows in LINEAR_SWEEP_ROWS if rows <= max_rows]


def infer_model_geometry(model_path: str) -> dict[str, int]:
    """从 HuggingFace config 推导 ModelRunner 所需的几何参数。"""
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    num_hidden_layers = getattr(cfg, "num_hidden_layers", None)
    num_attention_heads = getattr(cfg, "num_attention_heads", None)
    hidden_size = getattr(cfg, "hidden_size", None)
    if num_hidden_layers is None or num_attention_heads is None or hidden_size is None:
        raise RuntimeError(
            "模型 config 缺少 num_hidden_layers / num_attention_heads / hidden_size，"
            "无法自动构建 quant benchmark 的 EngineConfig"
        )

    num_kv_heads = getattr(cfg, "num_key_value_heads", num_attention_heads)
    head_dim = getattr(cfg, "head_dim", hidden_size // num_attention_heads)
    return {
        "num_hidden_layers": int(num_hidden_layers),
        "num_kv_heads": int(num_kv_heads),
        "head_dim": int(head_dim),
    }


def find_benchmark_linear(model: torch.nn.Module, mode: str) -> tuple[str, torch.nn.Module]:
    """选择最能代表当前 Phase 16 主线的线性层。

    优先选 MLP 的 gate/up/down_proj；fp16 选 nn.Linear，W8A8 选 QuantLinear。
    """
    target_type = QuantLinear if mode == "w8a8" else nn.Linear
    candidates: list[tuple[str, torch.nn.Module]] = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, target_type)
    ]
    if not candidates:
        raise RuntimeError(f"未找到适合 mode={mode!r} 的 benchmark linear 模块")

    preferred_suffixes = ("gate_proj", "up_proj", "down_proj")
    for suffix in preferred_suffixes:
        for name, module in candidates:
            if name.endswith(suffix):
                return name, module
    return candidates[0]


def run_linear_sweep(
    runner: ModelRunner,
    mode: str,
    warmup_iters: int = 10,
    bench_iters: int = 50,
    sweep_rows: Optional[list[int]] = None,
) -> dict:
    """对单个代表性线性层做 M-sweep，回答 _int_mm 何时才会命中。"""
    if sweep_rows is None:
        sweep_rows = build_linear_sweep_rows()

    layer_name, module = find_benchmark_linear(runner.model, mode)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[runner.config.dtype]
    device = runner.config.device
    rows_results = []

    for rows in sweep_rows:
        x = torch.randn(rows, module.in_features, device=device, dtype=dtype)
        with torch.inference_mode():
            if mode == "w8a8":
                QuantLinear.reset_runtime_stats()
            for _ in range(warmup_iters):
                _ = module(x)
            _sync_cuda()
            if mode == "w8a8":
                QuantLinear.reset_runtime_stats()
            t0 = time.perf_counter()
            for _ in range(bench_iters):
                _ = module(x)
            _sync_cuda()
            t1 = time.perf_counter()

        quant_stats = QuantLinear.get_runtime_stats() if mode == "w8a8" else None
        rows_results.append(
            {
                "rows": rows,
                "latency_ms": 1000.0 * (t1 - t0) / bench_iters,
                "quant_stats": quant_stats,
            }
        )

    return {
        "mode": mode,
        "layer_name": layer_name,
        "in_features": module.in_features,
        "out_features": module.out_features,
        "rows": rows_results,
    }


def build_runner(
    model_path: str,
    mode: str,
    batch_size: int,
    num_gpu_blocks: int = 200,
) -> tuple[EngineConfig, KVCacheManager, ModelRunner]:
    """构建用于 engine_compare 与 linear_sweep 共享的 runner。"""
    geometry = infer_model_geometry(model_path)
    cfg = EngineConfig(
        model_name=model_path,
        device="cuda:0",
        dtype="float16",
        max_batch_size=batch_size,
        max_model_len=512,
        block_size=256,
        num_gpu_blocks=num_gpu_blocks,
        num_hidden_layers=geometry["num_hidden_layers"],
        num_kv_heads=geometry["num_kv_heads"],
        head_dim=geometry["head_dim"],
        quant_mode="w8a8" if mode == "w8a8" else "",
    )

    kv_cache = KVCacheManager(cfg)
    runner = ModelRunner(cfg, kv_cache)
    return cfg, kv_cache, runner


def run_benchmark(
    runner: ModelRunner,
    kv_cache: KVCacheManager,
    mode: str,
    batch_size: int,
    max_new_tokens: int,
    warmup_iters: int = 2,
    bench_iters: int = 5,
    reference_token_ids: Optional[list[list[int]]] = None,
) -> dict:
    """运行 engine compare benchmark，返回结果字典。"""
    weight_mem_mb = measure_weight_memory(runner.model)
    prompt_batches = build_prompt_batches(batch_size)
    prompt_count = sum(len(prompt_batch) for prompt_batch in prompt_batches)

    def run_one_batch() -> dict:
        params = SamplingParams(max_new_tokens=max_new_tokens, temperature=0.0)
        prefill_total = 0.0
        decode_total = 0.0
        generated_tokens_total = 0
        decode_generated_tokens_total = 0
        prefill_quant_stats_total = None
        decode_quant_stats_total = None
        output_token_ids_total: list[list[int]] = []
        output_texts_total: list[str] = []

        for batch_idx, prompt_batch in enumerate(prompt_batches):
            states: list[RequestState] = []
            try:
                for prompt_idx, prompt in enumerate(prompt_batch):
                    token_ids = runner.tokenizer.encode(prompt, add_special_tokens=True)
                    req = Request(
                        request_id=f"b{batch_idx}_r{prompt_idx}",
                        prompt=prompt,
                        sampling_params=params,
                    )
                    st = RequestState(request=req, prompt_token_ids=token_ids)
                    states.append(st)

                for st in states:
                    kv_cache.init_request(st)

                batch_prefill_quant_stats = None
                batch_decode_quant_stats = None

                if mode == "w8a8":
                    QuantLinear.reset_runtime_stats()
                _sync_cuda()
                t_prefill_start = time.perf_counter()
                runner.prefill(states)
                _sync_cuda()
                t_prefill_end = time.perf_counter()
                if mode == "w8a8":
                    batch_prefill_quant_stats = QuantLinear.get_runtime_stats()
                    prefill_quant_stats_total = _accumulate_quant_stats(
                        prefill_quant_stats_total,
                        batch_prefill_quant_stats,
                    )
                    QuantLinear.reset_runtime_stats()

                active = [s for s in states if not s.finished]
                _sync_cuda()
                t_decode_start = time.perf_counter()
                for _ in range(max_new_tokens - 1):
                    if not active:
                        break
                    runner.decode_batch(active)
                    active = [s for s in active if not s.finished]
                _sync_cuda()
                t_decode_end = time.perf_counter()
                if mode == "w8a8":
                    batch_decode_quant_stats = QuantLinear.get_runtime_stats()
                    decode_quant_stats_total = _accumulate_quant_stats(
                        decode_quant_stats_total,
                        batch_decode_quant_stats,
                    )

                output_token_ids = [list(st.generated_token_ids) for st in states]
                output_texts = [
                    runner.tokenizer.decode(token_ids, skip_special_tokens=True)
                    for token_ids in output_token_ids
                ]
                generated_tokens_total += sum(len(token_ids) for token_ids in output_token_ids)
                decode_generated_tokens_total += sum(
                    max(len(token_ids) - 1, 0) for token_ids in output_token_ids
                )
                output_token_ids_total.extend(output_token_ids)
                output_texts_total.extend(output_texts)
                prefill_total += t_prefill_end - t_prefill_start
                decode_total += t_decode_end - t_decode_start
            finally:
                for st in states:
                    kv_cache.free_request(st)

        return {
            "texts": output_texts_total,
            "token_ids": output_token_ids_total,
            "prefill_s": prefill_total,
            "decode_s": decode_total,
            "e2e_s": prefill_total + decode_total,
            "generated_tokens": generated_tokens_total,
            "decode_generated_tokens": decode_generated_tokens_total,
            "prefill_quant_stats": prefill_quant_stats_total,
            "decode_quant_stats": decode_quant_stats_total,
        }

    # warmup
    for _ in range(warmup_iters):
        run_one_batch()
        gc.collect()
        _sync_cuda()

    total_prefill_s = 0.0
    total_decode_s = 0.0
    total_e2e_s = 0.0
    total_generated_tokens = 0
    total_decode_generated_tokens = 0
    outputs = None
    output_token_ids = None
    prefill_quant_stats_total = None
    decode_quant_stats_total = None

    for _ in range(bench_iters):
        batch_result = run_one_batch()
        gc.collect()
        total_prefill_s += batch_result["prefill_s"]
        total_decode_s += batch_result["decode_s"]
        total_e2e_s += batch_result["e2e_s"]
        total_generated_tokens += batch_result["generated_tokens"]
        total_decode_generated_tokens += batch_result["decode_generated_tokens"]
        outputs = batch_result["texts"]
        output_token_ids = batch_result["token_ids"]
        prefill_quant_stats_total = _accumulate_quant_stats(
            prefill_quant_stats_total,
            batch_result["prefill_quant_stats"],
        )
        decode_quant_stats_total = _accumulate_quant_stats(
            decode_quant_stats_total,
            batch_result["decode_quant_stats"],
        )

    decode_throughput = (
        total_decode_generated_tokens / total_decode_s
        if total_decode_generated_tokens > 0 and total_decode_s > 0
        else None
    )
    decode_tpot_ms = (
        1000.0 * total_decode_s / total_decode_generated_tokens
        if total_decode_generated_tokens > 0
        else None
    )
    e2e_throughput = (
        total_generated_tokens / total_e2e_s
        if total_generated_tokens > 0 and total_e2e_s > 0
        else 0.0
    )

    token_match_rate: Optional[float] = None
    sequence_exact_rate: Optional[float] = None
    match_metrics = None
    if reference_token_ids is not None and output_token_ids is not None:
        match_metrics = compute_match_metrics(reference_token_ids, output_token_ids)
        token_match_rate = match_metrics["token_match_rate"]
        sequence_exact_rate = match_metrics["sequence_exact_rate"]

    result = {
        "mode": mode,
        "batch_size": batch_size,
        "prompt_count": prompt_count,
        "num_prompt_batches": len(prompt_batches),
        "max_new_tokens": max_new_tokens,
        "weight_mem_mb": weight_mem_mb,
        "avg_prefill_ms": 1000.0 * total_prefill_s / (bench_iters * len(prompt_batches)),
        "eval_prefill_ms": 1000.0 * total_prefill_s / bench_iters,
        "decode_tps": decode_throughput,
        "decode_tpot_ms": decode_tpot_ms,
        "e2e_tps": e2e_throughput,
        "token_match_rate": token_match_rate,
        "sequence_exact_rate": sequence_exact_rate,
        "match_metrics": match_metrics,
        "prefill_quant_stats": prefill_quant_stats_total,
        "decode_quant_stats": decode_quant_stats_total,
        "quant_contract": QuantLinear.get_contract() if mode == "w8a8" else None,
        "quant_compute_note": build_quant_compute_note(
            QuantLinear.get_contract() if mode == "w8a8" else None,
            decode_quant_stats_total,
        ),
        "outputs": outputs,
        "output_token_ids": output_token_ids,
    }
    return result


def print_result(r: dict) -> None:
    mode = r["mode"].upper()
    print(f"\n{'='*50}")
    print(f"Mode          : {mode}")
    print(f"Batch size    : {r['batch_size']}")
    print(f"Prompt count  : {r['prompt_count']} ({r['num_prompt_batches']} engine batches)")
    print(f"Max new tokens: {r['max_new_tokens']}")
    print(f"Weight memory : {r['weight_mem_mb']:.1f} MB")
    print(f"Avg prefill   : {r['avg_prefill_ms']:.2f} ms/engine batch")
    print(f"Eval prefill  : {r['eval_prefill_ms']:.2f} ms/full prompt set")
    if r["decode_tps"] is not None:
        print(f"Decode TPS    : {r['decode_tps']:.1f} tokens/s")
        print(f"Decode TPOT   : {r['decode_tpot_ms']:.2f} ms/token")
    else:
        print("Decode TPS    : N/A")
        print("Decode TPOT   : N/A")
    print(f"E2E TPS       : {r['e2e_tps']:.1f} tokens/s")
    if r["token_match_rate"] is not None:
        print(f"Token match   : {r['token_match_rate']*100:.1f}% vs fp16")
    if r["sequence_exact_rate"] is not None:
        print(f"Seq exact     : {r['sequence_exact_rate']*100:.1f}% vs fp16")
    if r["prefill_quant_stats"] is not None:
        print(f"Prefill quant : {_format_quant_stats(r['prefill_quant_stats'])}")
    if r["decode_quant_stats"] is not None:
        print(f"Decode quant  : {_format_quant_stats(r['decode_quant_stats'])}")
    if r.get("quant_compute_note") is not None:
        print(f"Quant note    : {r['quant_compute_note']}")
    print(f"{'='*50}")


def print_comparison(fp16: dict, w8a8: dict) -> None:
    print("\n" + "=" * 60)
    print("fp16 vs W8A8 对比")
    print("=" * 60)
    print(f"{'指标':<20} {'fp16':>12} {'W8A8':>12} {'比值':>10}")
    print("-" * 60)
    print(f"{'权重显存 (MB)':<20} {fp16['weight_mem_mb']:>12.1f} {w8a8['weight_mem_mb']:>12.1f} "
          f"{w8a8['weight_mem_mb']/fp16['weight_mem_mb']:>10.3f}×")
    if fp16["decode_tps"] is not None and w8a8["decode_tps"] is not None:
        print(f"{'Decode 吞吐':<20} {fp16['decode_tps']:>12.1f} {w8a8['decode_tps']:>12.1f} "
              f"{w8a8['decode_tps']/fp16['decode_tps']:>10.3f}×")
        print(f"{'Decode TPOT (ms)':<20} {fp16['decode_tpot_ms']:>12.2f} {w8a8['decode_tpot_ms']:>12.2f} "
              f"{w8a8['decode_tpot_ms']/fp16['decode_tpot_ms']:>10.3f}×")
    print(f"{'E2E 吞吐':<20} {fp16['e2e_tps']:>12.1f} {w8a8['e2e_tps']:>12.1f} "
          f"{w8a8['e2e_tps']/fp16['e2e_tps']:>10.3f}×")
    if w8a8["token_match_rate"] is not None:
        print(f"{'Token match':<20} {'—':>12} {w8a8['token_match_rate']*100:>11.1f}% {'—':>10}")
    if w8a8["sequence_exact_rate"] is not None:
        print(f"{'Seq exact':<20} {'—':>12} {w8a8['sequence_exact_rate']*100:>11.1f}% {'—':>10}")
    print("=" * 60)
    print("\n验收标准检查：")
    mem_ratio = w8a8['weight_mem_mb'] / fp16['weight_mem_mb']
    print(f"  权重显存降低 ≥ 30%: {'✓' if mem_ratio <= 0.70 else '✗'} (降低 {(1-mem_ratio)*100:.1f}%)")
    if w8a8["token_match_rate"] is not None:
        em = w8a8["token_match_rate"]
        print(f"  Greedy token match ≥ 70%: {'✓' if em >= 0.70 else '✗'} ({em*100:.1f}%)")
    if fp16["decode_tps"] is not None and w8a8["decode_tps"] is not None:
        thr_ratio = w8a8["decode_tps"] / fp16["decode_tps"]
        print(f"  Stretch: decode 吞吐 ≥ 1.10×: {'✓' if thr_ratio >= 1.10 else '✗'} ({thr_ratio:.3f}×)")
        if thr_ratio < 1.10 and w8a8["decode_quant_stats"] is not None:
            print(f"  Decode 量化路径: {_format_quant_stats(w8a8['decode_quant_stats'])}")
    if w8a8.get("quant_compute_note") is not None:
        print(f"  Quant compute note: {w8a8['quant_compute_note']}")


def print_linear_sweep(result: dict) -> None:
    """输出单模式 linear sweep 表。"""
    print("\n" + "=" * 60)
    print(f"Linear Sweep ({result['mode'].upper()})")
    print("=" * 60)
    print(f"Layer         : {result['layer_name']}")
    print(f"Shape         : {result['in_features']} -> {result['out_features']}")
    print(f"{'M(rows)':<10} {'Latency(ms)':>14} {'Quant Path':>28}")
    print("-" * 60)
    for row in result["rows"]:
        print(
            f"{row['rows']:<10} {row['latency_ms']:>14.4f} "
            f"{format_linear_quant_stats(row['quant_stats']):>28}"
        )


def print_linear_sweep_comparison(fp16: dict, w8a8: dict) -> None:
    """输出 fp16 vs W8A8 的 linear sweep 对比。"""
    fp16_rows = {row["rows"]: row for row in fp16["rows"]}
    w8a8_rows = {row["rows"]: row for row in w8a8["rows"]}

    print("\n" + "=" * 72)
    print("Linear Sweep Compare")
    print("=" * 72)
    print(f"Layer         : {fp16['layer_name']} / {w8a8['layer_name']}")
    print(f"Shape         : {fp16['in_features']} -> {fp16['out_features']}")
    print(f"{'M(rows)':<10} {'fp16(ms)':>12} {'W8A8(ms)':>12} {'比值':>10} {'W8A8 Path':>22}")
    print("-" * 72)
    for rows in sorted(fp16_rows):
        fp16_row = fp16_rows[rows]
        w8a8_row = w8a8_rows[rows]
        ratio = w8a8_row["latency_ms"] / fp16_row["latency_ms"]
        print(
            f"{rows:<10} {fp16_row['latency_ms']:>12.4f} {w8a8_row['latency_ms']:>12.4f} "
            f"{ratio:>10.3f}× {format_linear_quant_stats(w8a8_row['quant_stats']):>22}"
        )


def main():
    parser = argparse.ArgumentParser(description="W8A8 量化 benchmark")
    parser.add_argument("--model", required=True, help="本地模型路径")
    parser.add_argument("--mode", choices=["fp16", "w8a8"], default="fp16",
                        help="量化模式（单模式运行时使用）")
    parser.add_argument("--compare", action="store_true",
                        help="同时运行 fp16 和 W8A8 并输出对比表")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="每次 engine batch 的请求数上限；固定 12 条 prompt 会按该大小拆批")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-gpu-blocks", type=int, default=200)
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--bench-iters", type=int, default=5)
    parser.add_argument("--linear-warmup-iters", type=int, default=10)
    parser.add_argument("--linear-bench-iters", type=int, default=50)
    args = parser.parse_args()

    if args.compare:
        print("运行 fp16 基线...")
        _, fp16_kv, fp16_runner = build_runner(
            args.model, "fp16", args.batch_size, args.num_gpu_blocks
        )
        fp16_result = run_benchmark(
            fp16_runner, fp16_kv, "fp16", args.batch_size, args.max_new_tokens,
            args.warmup_iters, args.bench_iters,
        )
        print_result(fp16_result)
        fp16_linear = run_linear_sweep(
            fp16_runner,
            "fp16",
            warmup_iters=args.linear_warmup_iters,
            bench_iters=args.linear_bench_iters,
        )

        # 清理 GPU 显存
        del fp16_runner, fp16_kv
        gc.collect()
        _sync_cuda()
        torch.cuda.empty_cache()

        print("\n运行 W8A8 量化...")
        _, w8a8_kv, w8a8_runner = build_runner(
            args.model, "w8a8", args.batch_size, args.num_gpu_blocks
        )
        w8a8_result = run_benchmark(
            w8a8_runner, w8a8_kv, "w8a8", args.batch_size, args.max_new_tokens,
            args.warmup_iters, args.bench_iters,
            reference_token_ids=fp16_result["output_token_ids"],
        )
        print_result(w8a8_result)
        w8a8_linear = run_linear_sweep(
            w8a8_runner,
            "w8a8",
            warmup_iters=args.linear_warmup_iters,
            bench_iters=args.linear_bench_iters,
        )
        print_comparison(fp16_result, w8a8_result)
        print_linear_sweep_comparison(fp16_linear, w8a8_linear)
    else:
        _, kv_cache, runner = build_runner(
            args.model, args.mode, args.batch_size, args.num_gpu_blocks
        )
        result = run_benchmark(
            runner, kv_cache, args.mode, args.batch_size, args.max_new_tokens,
            args.warmup_iters, args.bench_iters,
        )
        print_result(result)
        linear_result = run_linear_sweep(
            runner,
            args.mode,
            warmup_iters=args.linear_warmup_iters,
            bench_iters=args.linear_bench_iters,
        )
        print_linear_sweep(linear_result)


if __name__ == "__main__":
    main()
