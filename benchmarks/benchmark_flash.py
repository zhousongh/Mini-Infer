"""
benchmarks/benchmark_flash.py — Phase 6 True PagedAttention Benchmark

对比 mini-infer Phase 6（flash_attn_with_kvcache + block_table）与 HuggingFace Transformers
baseline 的推理性能。

Phase 6 变化（相比 Phase 3 benchmark_mini.py）：
  - decode_batch 路径：消除 gather_batch_kv / write_decode_kv，改用 flash_attn 直接 block 寻址
  - block_size 必须为 256 的倍数（flash_attn kernel 约束），本脚本默认使用 256

测量指标：
  - Throughput（tokens/s）：所有请求输出 token 总数 / 总耗时
  - TTFT（ms）：第一个 token 产出延迟（单请求 prefill 时间近似）
  - TPOT（ms/tok）：首 token 后每 token 平均延迟
  - Peak Memory（GB）：CUDA 峰值显存占用

用法：
    HF_HUB_OFFLINE=1 python benchmarks/benchmark_flash.py \\
        --model /path/to/model \\
        --batch-size 8 \\
        --max-new-tokens 128

对比运行（mini-infer vs HF baseline）：
    HF_HUB_OFFLINE=1 python benchmarks/benchmark_flash.py --model /path/to/model --compare
"""

import argparse
import time
from dataclasses import dataclass

import torch

from mini_infer import EngineConfig, LLMEngine

# 与 benchmark_hf.py 保持一致的 prompt 集合
PROMPTS = [
    "请介绍一下大语言模型的推理优化技术。",
    "什么是 PagedAttention？它解决了什么问题？",
    "解释 KV Cache 在 Transformer 推理中的作用。",
    "Continuous Batching 和静态 Batching 的区别是什么？",
    "如何在多 GPU 上做张量并行推理？",
    "介绍一下 FlashAttention 的核心思想。",
    "大模型推理时显存占用的主要来源有哪些？",
    "什么是 Prefill 阶段和 Decode 阶段的区别？",
]


@dataclass
class BenchmarkResult:
    engine: str
    model_name: str
    batch_size: int
    max_new_tokens: int
    total_time_s: float
    throughput_tok_s: float
    ttft_ms: float
    tpot_ms: float
    peak_memory_gb: float


def benchmark_flash(
    model_name: str,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    device: str = "cuda:0",
    dtype: str = "float16",
    num_gpu_blocks: int = 512,
    # Qwen2.5-7B-Instruct 默认架构参数
    num_hidden_layers: int = 28,
    num_kv_heads: int = 4,
    head_dim: int = 128,
) -> BenchmarkResult:
    """Phase 6 mini-infer benchmark（True PagedAttention）。"""
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("benchmark_flash 需要可用的 CUDA GPU，但当前环境未检测到。")

    config = EngineConfig(
        model_name=model_name,
        device=device,
        dtype=dtype,
        max_batch_size=batch_size,
        block_size=256,  # flash_attn kernel 约束：block_size 必须是 256 的倍数
        num_gpu_blocks=num_gpu_blocks,
        num_hidden_layers=num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    print(f"[flash] 加载 mini-infer Phase 6: {model_name}")
    engine = LLMEngine(config)
    prompts = PROMPTS[:batch_size]
    if batch_size > len(PROMPTS):
        print(f"[警告] batch_size={batch_size} > 可用 prompt 数={len(PROMPTS)}，"
              f"实际使用 {len(PROMPTS)} 个 prompt。")

    # 预热
    print("[flash] 预热中（1 条请求，4 token）...")
    engine.generate(prompts[:1], max_new_tokens=4)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    # 测量 TTFT（单请求 prefill 时间近似）
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t_ttft = time.perf_counter()
    engine.generate(prompts[:1], max_new_tokens=1)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    ttft_ms = (time.perf_counter() - t_ttft) * 1000

    # 测量整体 throughput
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    outputs = engine.generate(prompts, max_new_tokens=max_new_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    total_time = time.perf_counter() - t0

    total_tokens = sum(
        len(engine.model_runner.tokenizer.encode(o, add_special_tokens=False))
        for o in outputs
    )
    throughput = total_tokens / total_time

    decode_tokens = max(total_tokens - batch_size, 1)
    decode_time_s = max(total_time - ttft_ms / 1000, 1e-6)
    tpot_ms = decode_time_s / decode_tokens * 1000

    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0

    return BenchmarkResult(
        engine="mini-infer-phase6",
        model_name=model_name,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        total_time_s=total_time,
        throughput_tok_s=throughput,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        peak_memory_gb=peak_mem_gb,
    )


def benchmark_hf_baseline(
    model_name: str,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    device: str = "cuda:1",
    dtype: str = "float16",
) -> BenchmarkResult:
    """HuggingFace Transformers baseline（与 benchmark_hf.py 逻辑一致，内联实现避免跨文件依赖）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("HF baseline 需要可用的 CUDA GPU。")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[dtype]

    print(f"[hf] 加载 HF baseline: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, trust_remote_code=True, device_map=device
    )
    model.eval()

    prompts = PROMPTS[:batch_size]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

    # 预热
    print("[hf] 预热中...")
    with torch.no_grad():
        model.generate(**{k: v[:1] for k, v in inputs.items()}, max_new_tokens=4)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)

    # TTFT（单请求）
    single_inputs = {k: v[:1] for k, v in inputs.items()}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t_ttft = time.perf_counter()
    with torch.no_grad():
        model.generate(**single_inputs, max_new_tokens=1)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    ttft_ms = (time.perf_counter() - t_ttft) * 1000

    # Throughput
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
    total_time = time.perf_counter() - t0

    prompt_len = inputs["input_ids"].shape[1]
    total_tokens = sum(len(o) - prompt_len for o in outputs)
    throughput = total_tokens / total_time

    decode_tokens = max(total_tokens - batch_size, 1)
    decode_time_s = max(total_time - ttft_ms / 1000, 1e-6)
    tpot_ms = decode_time_s / decode_tokens * 1000

    peak_mem_gb = torch.cuda.max_memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0.0

    del model
    torch.cuda.empty_cache()

    return BenchmarkResult(
        engine="hf-transformers",
        model_name=model_name,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        total_time_s=total_time,
        throughput_tok_s=throughput,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        peak_memory_gb=peak_mem_gb,
    )


def print_result(result: BenchmarkResult) -> None:
    print(f"\n{'=' * 55}")
    print(f"  引擎:           {result.engine}")
    print(f"  模型:           {result.model_name}")
    print(f"  batch_size:     {result.batch_size}")
    print(f"  max_new_tokens: {result.max_new_tokens}")
    print(f"  总耗时:         {result.total_time_s:.2f} s")
    print(f"  Throughput:     {result.throughput_tok_s:.1f} tokens/s")
    print(f"  TTFT（近似）:   {result.ttft_ms:.1f} ms")
    print(f"  TPOT（近似）:   {result.tpot_ms:.2f} ms/tok")
    print(f"  Peak Mem:       {result.peak_memory_gb:.2f} GB")
    print(f"{'=' * 55}\n")


def print_comparison(mini: BenchmarkResult, hf: BenchmarkResult) -> None:
    ratio = mini.throughput_tok_s / hf.throughput_tok_s * 100
    print(f"\n{'=' * 65}")
    print("  Phase 6 True PagedAttention vs HF Transformers 对比")
    print(f"{'=' * 65}")
    print(f"  {'指标':<20} {'mini-infer Phase 6':>20} {'HF baseline':>20}")
    print(f"  {'-' * 62}")
    print(f"  {'Throughput (tok/s)':<20} {mini.throughput_tok_s:>20.1f} {hf.throughput_tok_s:>20.1f}")
    print(f"  {'TTFT (ms)':<20} {mini.ttft_ms:>20.1f} {hf.ttft_ms:>20.1f}")
    print(f"  {'TPOT (ms/tok)':<20} {mini.tpot_ms:>20.2f} {hf.tpot_ms:>20.2f}")
    print(f"  {'Peak Mem (GB)':<20} {mini.peak_memory_gb:>20.2f} {hf.peak_memory_gb:>20.2f}")
    print(f"  {'-' * 62}")
    print(f"  mini-infer throughput = {ratio:.1f}% of HF baseline")
    print(f"{'=' * 65}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-infer Phase 6 benchmark（True PagedAttention）")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--hf-device", type=str, default="cuda:1", help="HF baseline 使用的 GPU（--compare 模式）")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--num-gpu-blocks", type=int, default=512)
    parser.add_argument("--compare", action="store_true", help="同时跑 HF baseline 并对比（需要 2 个 GPU）")
    args = parser.parse_args()

    mini_result = benchmark_flash(
        model_name=args.model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        num_gpu_blocks=args.num_gpu_blocks,
    )
    print_result(mini_result)

    if args.compare:
        if torch.cuda.device_count() < 2:
            print("[警告] --compare 需要至少 2 个 GPU，当前只有 1 个，跳过 HF baseline。")
            return
        hf_result = benchmark_hf_baseline(
            model_name=args.model,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device=args.hf_device,
            dtype=args.dtype,
        )
        print_result(hf_result)
        print_comparison(mini_result, hf_result)


if __name__ == "__main__":
    main()
