"""这个文件实现 mini-infer 引擎的 benchmark，对标 benchmark_hf.py 的同款 prompt 和指标。
当前阶段（Phase 6）：True PagedAttention（flash_attn_with_kvcache + block_table），
  消除 gather_batch_kv / write_decode_kv，block_size 必须为 256 的倍数（flash_attn 约束）。
运行环境：默认在 Ubuntu 项目环境中执行；如使用 CUDA 设备，需要已就绪的 GPU 和模型权重。

测量指标：
  - Throughput（tokens/s）：所有请求输出 token 总数 / 总耗时
  - TTFT（Time To First Token，ms）：第一个 token 产出的延迟，对应 prefill 时间
  - TPOT（Time Per Output Token，ms/tok）：首 token 之后每个 token 的平均延迟
  - Peak Memory（GB）：CUDA 峰值显存占用

新增（Phase 3）：
  - --mixed 模式：使用长短不一的 prompt，体现 continuous batching 在混合长度场景下的行为
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
    engine_phase: str
    model_name: str
    batch_size: int
    max_new_tokens: int
    total_time_s: float
    throughput_tok_s: float
    ttft_ms: float          # 首 token 时间（仅 batch=1 时有意义，此处用 prefill 时间近似）
    tpot_ms: float          # 首 token 后每 token 延迟
    peak_memory_gb: float


def benchmark_mini(
    model_name: str,
    batch_size: int = 4,
    max_new_tokens: int = 128,
    device: str = "cuda:0",
    dtype: str = "float16",
    num_gpu_blocks: int = 512,
    # Qwen2.5-7B-Instruct 模型架构参数（num_key_value_heads=4，28 Q heads GQA）
    num_hidden_layers: int = 28,
    num_kv_heads: int = 4,
    head_dim: int = 128,
) -> BenchmarkResult:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("benchmark_mini 需要可用的 CUDA GPU，但当前环境未检测到。")

    config = EngineConfig(
        model_name=model_name,
        device=device,
        dtype=dtype,
        max_batch_size=batch_size,
        block_size=256,
        num_gpu_blocks=num_gpu_blocks,
        num_hidden_layers=num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    print(f"初始化 LLMEngine (Phase 6): {model_name}")
    engine = LLMEngine(config)

    prompts = PROMPTS[:batch_size]

    print("热身中（1 条请求，4 token）...")
    _ = engine.generate(prompts[:1], max_new_tokens=4)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # ── 测量 TTFT（用单条请求的 prefill 时间近似）──
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_ttft_start = time.perf_counter()
    _ = engine.generate(prompts[:1], max_new_tokens=1)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t_ttft_start) * 1000

    # ── 测量整体 throughput 和 TPOT ──
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    outputs = engine.generate(prompts, max_new_tokens=max_new_tokens)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    total_tokens = sum(
        len(engine.model_runner.tokenizer.encode(o, add_special_tokens=False))
        for o in outputs
    )
    throughput = total_tokens / total_time

    # TPOT 近似计算说明：
    #   ttft_ms 来自单请求 prefill（batch=1, max_new_tokens=1），用作 prefill 时间代理。
    #   batch_size > 1 时，实际 prefill 时间与 ttft_ms 存在差异，TPOT 为近似值。
    #   decode_tokens = 总输出 token 数 - batch_size 个首 token（首 token 归入 prefill）
    decode_tokens = max(total_tokens - batch_size, 1)
    decode_time_s = max(total_time - ttft_ms / 1000, 1e-6)
    tpot_ms = decode_time_s / decode_tokens * 1000

    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    return BenchmarkResult(
        engine_phase="mini-infer-phase6",
        model_name=model_name,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        total_time_s=total_time,
        throughput_tok_s=throughput,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        peak_memory_gb=peak_mem_gb,
    )


# 混合长度 prompt 集合：长短不一，max_new_tokens 差异大，用于体现 continuous batching 的优势
# 格式：(prompt, max_new_tokens)
MIXED_WORKLOAD = [
    ("一句话介绍 KV Cache。", 32),
    ("详细解释 PagedAttention 的完整设计思路，包括 BlockTable、FreeBlockPool 和 batch decode 的实现细节，以及与传统 KV cache 的对比。", 256),
    ("GQA 是什么？", 32),
    ("从头实现一个支持 Continuous Batching 的推理引擎需要哪些核心模块？请逐一说明每个模块的职责和接口设计。", 256),
    ("FlashAttention 解决了什么问题？", 32),
    ("大模型推理中显存占用的来源有哪些？Paged KV Cache 如何减少显存碎片？请结合实际工程实现来解释。", 256),
    ("什么是 TTFT？", 32),
    ("解释 Prefill 和 Decode 阶段在计算特征上的本质区别，以及为什么它们需要不同的优化策略。", 256),
]


@dataclass
class MixedBenchmarkResult:
    engine_phase: str
    model_name: str
    num_requests: int
    total_time_s: float
    throughput_tok_s: float
    peak_memory_gb: float
    short_prompt_count: int   # prompt 较短的请求数（设计意图 max_new_tokens=32，实际统一用 max_tokens）
    long_prompt_count: int    # prompt 较长的请求数（设计意图 max_new_tokens=256）
    actual_max_new_tokens: int  # 实际传给 generate() 的 max_new_tokens（所有请求统一值）


def benchmark_mini_mixed(
    model_name: str,
    device: str = "cuda:0",
    dtype: str = "float16",
    num_gpu_blocks: int = 1024,
    num_hidden_layers: int = 28,
    num_kv_heads: int = 4,
    head_dim: int = 128,
) -> MixedBenchmarkResult:
    """
    混合长度 benchmark：长短 prompt 混合提交，max_new_tokens 差异大（32 vs 256）。
    体现 continuous batching 在异质请求下的调度行为：短请求早完成，KV 块及时归还。
    """
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("混合 benchmark 需要可用的 CUDA GPU。")

    max_batch = len(MIXED_WORKLOAD)
    config = EngineConfig(
        model_name=model_name,
        device=device,
        dtype=dtype,
        max_batch_size=max_batch,
        block_size=256,
        num_gpu_blocks=num_gpu_blocks,
        num_hidden_layers=num_hidden_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
    )

    print(f"初始化 LLMEngine（混合长度）: {model_name}")
    engine = LLMEngine(config)

    prompts = [p for p, _ in MIXED_WORKLOAD]
    max_tokens_list = [t for _, t in MIXED_WORKLOAD]
    short_count = sum(1 for t in max_tokens_list if t <= 32)
    long_count = len(max_tokens_list) - short_count

    print("热身中（1 条短请求，4 token）...")
    _ = engine.generate(prompts[:1], max_new_tokens=4)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # 混合提交：每条请求使用各自的 max_new_tokens
    # generate() 目前不支持 per-request max_new_tokens，用最大值运行全批
    # 短请求因 EOS 或较早完成而先结束，continuous batching 会及时释放其 KV 块
    max_new_tokens_all = max(max_tokens_list)
    print(f"运行混合 benchmark（{short_count} 短 + {long_count} 长，max_new_tokens={max_new_tokens_all}）...")

    t0 = time.perf_counter()
    outputs = engine.generate(prompts, max_new_tokens=max_new_tokens_all)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    total_tokens = sum(
        len(engine.model_runner.tokenizer.encode(o, add_special_tokens=False))
        for o in outputs
    )
    throughput = total_tokens / total_time
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    return MixedBenchmarkResult(
        engine_phase="mini-infer-phase6-mixed",
        model_name=model_name,
        num_requests=len(prompts),
        total_time_s=total_time,
        throughput_tok_s=throughput,
        peak_memory_gb=peak_mem_gb,
        short_prompt_count=short_count,
        long_prompt_count=long_count,
        actual_max_new_tokens=max_new_tokens_all,
    )


def print_result(result: BenchmarkResult) -> None:
    print("\n========== mini-infer Phase 6 Benchmark ==========")
    print(f"引擎:           {result.engine_phase}")
    print(f"模型:           {result.model_name}")
    print(f"batch_size:     {result.batch_size}")
    print(f"max_new_tokens: {result.max_new_tokens}")
    print(f"总耗时:         {result.total_time_s:.2f} s")
    print(f"Throughput:     {result.throughput_tok_s:.1f} tokens/s")
    print(f"TTFT（近似）:   {result.ttft_ms:.1f} ms")
    print(f"TPOT（近似）:   {result.tpot_ms:.2f} ms/tok")
    print(f"Peak Mem:       {result.peak_memory_gb:.2f} GB")
    print("===================================================\n")


def print_mixed_result(result: MixedBenchmarkResult) -> None:
    print("\n========== mini-infer Phase 6 Mixed Benchmark ==========")
    print(f"引擎:           {result.engine_phase}")
    print(f"模型:           {result.model_name}")
    print(f"请求数:         {result.num_requests}（短 prompt {result.short_prompt_count} + 长 prompt {result.long_prompt_count}）")
    print(f"actual max_new_tokens: {result.actual_max_new_tokens}（所有请求统一值，短 prompt 不提前截断）")
    print(f"总耗时:         {result.total_time_s:.2f} s")
    print(f"Throughput:     {result.throughput_tok_s:.1f} tokens/s")
    print(f"Peak Mem:       {result.peak_memory_gb:.2f} GB")
    print("=========================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-infer Phase 6 benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--num-gpu-blocks", type=int, default=512)
    parser.add_argument("--mixed", action="store_true", help="运行混合长度 benchmark")
    args = parser.parse_args()

    if args.mixed:
        result = benchmark_mini_mixed(
            model_name=args.model,
            device=args.device,
            dtype=args.dtype,
            num_gpu_blocks=args.num_gpu_blocks,
        )
        print_mixed_result(result)
    else:
        result = benchmark_mini(
            model_name=args.model,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            dtype=args.dtype,
            num_gpu_blocks=args.num_gpu_blocks,
        )
        print_result(result)


if __name__ == "__main__":
    main()
