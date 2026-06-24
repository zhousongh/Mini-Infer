"""
Phase 13 Tensor Parallel benchmark。

对比三种配置的吞吐（tok/s）和每卡峰值显存（GB）：
  single  — 单卡推理（LLMEngine 或直接 HF model.generate）
  tp=2    — Tensor Parallel TP=2（TPEngine，NCCL all-reduce）
  pp      — Pipeline Parallel（PPEngine，device_map="balanced"）

TP 模式需要通过 torchrun 启动（两种方式均支持）：
  # 方式 1：直接调用（--mode single / --mode pp）
  python benchmarks/benchmark_tp.py --model /path/to/model

  # 方式 2：TPEngine 内部 mp.spawn（--mode tp）
  python benchmarks/benchmark_tp.py --model /path/to/model --mode tp

  # 方式 3：torchrun 直接启动 TP worker（--mode torchrun_tp）
  torchrun --nproc_per_node 2 benchmarks/benchmark_tp.py --model /path/to/model --mode torchrun_tp

测量口径说明：
  - throughput = total_tokens / wall_clock_seconds
  - peak_vram  = torch.cuda.max_memory_allocated() 在推理结束后测量
  - 未测量 TTFT / TPOT（不接入 LLMEngine，不适用）
  - `--warmup` / `--runs` 可配置，默认 warmup=2、runs=5
  - `--mode tp` 包含 mp.spawn + 模型加载开销，仅适合功能验证；真实 TP 吞吐请用 `--mode torchrun_tp`

依赖：
  torch.distributed / NCCL（TP 模式），PPEngine（PP 模式），transformers

环境：
  RTX 4090 ×2，PyTorch 2.1.2+cu121，transformers 4.43.4
"""

from __future__ import annotations

import argparse
import os
import time

import torch

# ──────────────────────────────────────────────
# 工具
# ──────────────────────────────────────────────


def _tok_count(tok, texts: list[str]) -> int:
    total = 0
    for t in texts:
        total += len(tok.encode(t, add_special_tokens=False))
    return total


def _vram_gb(device: str | int = 0) -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1e9


# ──────────────────────────────────────────────
# 单卡推理（standard HF generate）
# ──────────────────────────────────────────────


def _bench_single(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int,
    warmup: int,
    runs: int,
    dtype: str,
) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype={"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype],
        trust_remote_code=True,
        device_map="cuda:0",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    torch.cuda.reset_peak_memory_stats("cuda:0")

    def _run():
        texts = []
        for p in prompts:
            inp = tok(p, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                out = model.generate(
                    **inp,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=50,
                    pad_token_id=tok.pad_token_id,
                )
            texts.append(tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True))
        return texts

    for _ in range(warmup):
        _run()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        result = _run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_tok = _tok_count(tok, result) * runs
    throughput = total_tok / elapsed
    vram = _vram_gb("cuda:0")

    return {
        "mode": "single",
        "throughput_tok_s": throughput,
        "vram_gpu0_gb": vram,
        "vram_gpu1_gb": 0.0,
        "sample_output": result[0][:60],
    }



# ──────────────────────────────────────────────
# Tensor Parallel（TPEngine，mp.spawn 内部）
# ──────────────────────────────────────────────


def _bench_tp(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int,
    warmup: int,
    runs: int,
    dtype: str,
    tp_size: int = 2,
) -> dict:
    from transformers import AutoTokenizer

    from mini_infer.parallel.tp_engine import TPEngine

    engine = TPEngine(model_path=model_path, tp_size=tp_size, dtype=dtype)
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    for _ in range(warmup):
        result = engine.generate(prompts, max_new_tokens=max_new_tokens)

    t0 = time.perf_counter()
    for _ in range(runs):
        result = engine.generate(prompts, max_new_tokens=max_new_tokens)
    elapsed = time.perf_counter() - t0

    total_tok = _tok_count(tok, result) * runs
    throughput = total_tok / elapsed

    return {
        "mode": f"tp={tp_size}",
        "throughput_tok_s": throughput,
        "vram_gpu0_gb": 0.0,    # 在子进程内测量，主进程不可得 → N/A
        "vram_gpu1_gb": 0.0,    # 同上
        "note_vram": "N/A (measured inside worker subprocess, see worker output)",
        "sample_output": result[0][:60],
    }


# ──────────────────────────────────────────────
# torchrun 直接 TP worker（--mode torchrun_tp）
# ──────────────────────────────────────────────


def _run_torchrun_tp(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int,
    warmup: int,
    runs: int,
    dtype: str,
) -> None:
    """
    在 torchrun 环境中直接运行 TP worker。
    每个进程初始化 dist，加载 TensorParallelModelRunner，
    Rank 0 打印 throughput + per-GPU VRAM。
    """
    import torch.distributed as dist

    from mini_infer.parallel.tp_model_runner import TensorParallelModelRunner

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    tp_size = dist.get_world_size()
    device = f"cuda:{rank}"

    torch.cuda.set_device(rank)  # 显式初始化 CUDA 设备，确保后续 memory stats API 可用
    torch.cuda.reset_peak_memory_stats(device)

    runner = TensorParallelModelRunner(
        model_path=model_path,
        rank=rank,
        tp_size=tp_size,
        device=device,
        dtype=dtype,
    )

    # Warmup
    for _ in range(warmup):
        runner.generate(prompts, max_new_tokens=max_new_tokens)

    # Measure
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(runs):
        result = runner.generate(prompts, max_new_tokens=max_new_tokens)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    # Peak VRAM for this rank
    vram = _vram_gb(device)

    if rank == 0:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        total_tok = _tok_count(tok, result) * runs
        throughput = total_tok / elapsed

        print(f"\n=== TP={tp_size} (torchrun mode) ===")
        print(f"  throughput:  {throughput:.1f} tok/s")
        print(f"  peak VRAM GPU0: {vram:.2f} GB")
        print(f"  sample:      {result[0][:60]!r}")

    # Collect VRAM from all ranks
    vram_tensor = torch.tensor([vram], device=device)
    all_vram = [torch.zeros_like(vram_tensor) for _ in range(tp_size)]
    dist.all_gather(all_vram, vram_tensor)

    if rank == 0:
        for r, v in enumerate(all_vram):
            print(f"  peak VRAM GPU{r}: {v.item():.2f} GB")

    dist.destroy_process_group()


# ──────────────────────────────────────────────
# PP benchmark helper（HF device_map='balanced'）
# ──────────────────────────────────────────────


def _get_pp_generate(model_path: str, dtype: str, max_new_tokens: int):
    """构建 PP generate 函数，返回 (generate_fn, tokenizer)。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_map[dtype],
        trust_remote_code=True,
        device_map="balanced",   # PP: balanced across 2 GPUs
        attn_implementation="flash_attention_2",
    )
    model.eval()

    def generate(prompts):
        results = []
        for p in prompts:
            inp = tok(p, return_tensors="pt").to("cuda:0")
            with torch.no_grad():
                out = model.generate(
                    **inp,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=50,
                    pad_token_id=tok.pad_token_id,
                )
            results.append(tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True))
        return results

    return generate, tok


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 13 TP benchmark")
    parser.add_argument(
        "--model",
        default=os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct"
            "/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
        ),
        help="模型本地路径",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "tp", "pp", "all", "torchrun_tp"],
        default="all",
        help="测试模式",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    PROMPTS = [
        "你好，请详细介绍一下量子计算的基本原理。",
        "Explain the difference between machine learning and deep learning.",
        "写一首关于秋天的五言绝句。",
    ]

    if args.mode == "torchrun_tp":
        _run_torchrun_tp(
            args.model, PROMPTS, args.max_new_tokens, args.warmup, args.runs, args.dtype
        )
        return

    results = []

    if args.mode in ("single", "all"):
        print("Running single GPU benchmark...")
        r = _bench_single(
            args.model, PROMPTS, args.max_new_tokens, args.warmup, args.runs, args.dtype
        )
        results.append(r)

    if args.mode == "tp":
        # 注意：--mode tp 通过 mp.spawn 实现，每次 generate() 都会重新启动进程并加载模型。
        # 计时包含进程启动 + 完整模型加载开销，不代表纯推理吞吐，仅适合功能验证。
        # 如需真实 TP 吞吐数字，请使用 --mode torchrun_tp（torchrun 模式不含加载开销）。
        print("Running TP=2 benchmark (mp.spawn, includes model load time)...")
        print("  WARNING: timing includes process spawn + model load. Use --mode torchrun_tp for throughput.")
        r = _bench_tp(
            args.model, PROMPTS, args.max_new_tokens,
            warmup=args.warmup, runs=args.runs,
            dtype=args.dtype,
        )
        results.append(r)

    if args.mode in ("pp", "all"):
        print("Running PP benchmark (device_map=balanced)...")
        r = _bench_pp_direct(
            args.model, PROMPTS, args.max_new_tokens, args.warmup, args.runs, args.dtype
        )
        results.append(r)

    # ── 打印结果表 ──
    print("\n" + "=" * 70)
    print(f"{'Mode':<12} {'Throughput':>14} {'VRAM GPU0':>12} {'VRAM GPU1':>12}")
    print("-" * 70)
    for r in results:
        tp_s = r.get("throughput_tok_s", 0)
        v0 = r.get("vram_gpu0_gb", 0)
        v1 = r.get("vram_gpu1_gb", 0)
        vram_str0 = f"{v0:.2f} GB" if v0 > 0 else "N/A"
        vram_str1 = f"{v1:.2f} GB" if v1 > 0 else "N/A"
        print(f"{r['mode']:<12} {tp_s:>12.1f}/s {vram_str0:>12} {vram_str1:>12}")
    print("=" * 70)

    for r in results:
        print(f"[{r['mode']}] sample: {r['sample_output']!r}")


def _bench_pp_direct(
    model_path: str,
    prompts: list[str],
    max_new_tokens: int,
    warmup: int,
    runs: int,
    dtype: str,
) -> dict:
    """PP benchmark：使用 HF device_map='balanced'（直接，不需要 PPEngine 适配）。"""
    generate, tok = _get_pp_generate(model_path, dtype, max_new_tokens)
    torch.cuda.reset_peak_memory_stats("cuda:0")
    torch.cuda.reset_peak_memory_stats("cuda:1")

    for _ in range(warmup):
        result = generate(prompts)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        result = generate(prompts)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_tok = _tok_count(tok, result) * runs
    throughput = total_tok / elapsed

    return {
        "mode": "pp",
        "throughput_tok_s": throughput,
        "vram_gpu0_gb": _vram_gb("cuda:0"),
        "vram_gpu1_gb": _vram_gb("cuda:1"),
        "sample_output": result[0][:60],
    }


if __name__ == "__main__":
    main()
