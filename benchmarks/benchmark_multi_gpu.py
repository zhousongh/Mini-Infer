"""
Phase 4 双卡 benchmark。对比三种推理配置在 2 × RTX 4090 上的性能：
  - single：单卡 LLMEngine（当前主线 paged decode 路径，cuda:0）
  - replica：双卡 ReplicaEngine（cuda:0 + cuda:1，数据并行）
  - pp：双卡 PPEngine（device_map="balanced"，HF Pipeline Parallel）
    注：PP = Pipeline Parallel（不同层在不同 GPU），不是 Tensor Parallel（同层 all-reduce）
    tp2 是 pp 的旧别名，两者等价。

测量指标：
  - Throughput（tok/s）：所有请求输出 token 总数 / 总耗时
  - TTFT（ms）：单请求首 token 延迟（prefill 近似），仅 single/replica 有意义
  - Peak Mem / GPU（GB）：各 GPU 峰值显存

运行方式：
  python benchmarks/benchmark_multi_gpu.py --mode single  --model /path/to/model
  python benchmarks/benchmark_multi_gpu.py --mode replica --model /path/to/model
  python benchmarks/benchmark_multi_gpu.py --mode pp      --model /path/to/model
"""

import argparse
import time
from dataclasses import dataclass

import torch

from mini_infer import EngineConfig, LLMEngine
from mini_infer.parallel.pp_engine import PPEngine
from mini_infer.parallel.replica_engine import ReplicaEngine

# 与 benchmark_mini.py 保持一致的 prompt 集合（8 条）
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

# Qwen2.5-7B-Instruct 架构参数
QWEN_LAYERS = 28
QWEN_KV_HEADS = 4
QWEN_HEAD_DIM = 128


@dataclass
class MultiGPUBenchmarkResult:
    mode: str          # "single" / "replica" / "pp"
    model_name: str
    batch_size: int
    max_new_tokens: int
    total_time_s: float
    throughput_tok_s: float
    ttft_ms: float
    # single/replica：mini-infer engine.generate(max_new_tokens=1) 的首 token 延迟
    # pp：HF model.generate(max_new_tokens=1) 的端到端延迟，含 PP 层间传输和 HF sampling 开销
    # 两种口径可比较趋势，但 pp 通常偏高（HF overhead），不宜直接数值对比
    peak_mem_gpu0_gb: float
    peak_mem_gpu1_gb: float


def _count_tokens(tokenizer, texts: list[str]) -> int:
    return sum(
        len(tokenizer.encode(t, add_special_tokens=False)) for t in texts
    )


def _reset_peak_memory() -> None:
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)


def _sync_all() -> None:
    for i in range(torch.cuda.device_count()):
        if torch.cuda.is_available():
            torch.cuda.synchronize(i)


def _peak_mem_gb(device_idx: int) -> float:
    if torch.cuda.is_available() and device_idx < torch.cuda.device_count():
        return torch.cuda.max_memory_allocated(device_idx) / 1e9
    return 0.0


# ── Single GPU benchmark ────────────────────────────────────────────────────

def benchmark_single(
    model_name: str,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    device: str = "cuda:0",
    dtype: str = "float16",
    num_gpu_blocks: int = 200,
) -> MultiGPUBenchmarkResult:
    config = EngineConfig(
        model_name=model_name,
        device=device,
        dtype=dtype,
        max_batch_size=batch_size,
        block_size=256,
        num_gpu_blocks=num_gpu_blocks,
        num_hidden_layers=QWEN_LAYERS,
        num_kv_heads=QWEN_KV_HEADS,
        head_dim=QWEN_HEAD_DIM,
    )
    print(f"[single] 初始化 LLMEngine: {model_name}")
    engine = LLMEngine(config)
    prompts = PROMPTS[:batch_size]

    print("[single] 热身（1 条请求，4 token）...")
    _ = engine.generate(prompts[:1], max_new_tokens=4)
    _sync_all()

    # TTFT
    _reset_peak_memory()
    t_ttft = time.perf_counter()
    _ = engine.generate(prompts[:1], max_new_tokens=1)
    _sync_all()
    ttft_ms = (time.perf_counter() - t_ttft) * 1000

    # 主 benchmark
    _reset_peak_memory()
    t0 = time.perf_counter()
    outputs = engine.generate(prompts, max_new_tokens=max_new_tokens)
    _sync_all()
    total_time = time.perf_counter() - t0

    total_tokens = _count_tokens(engine.model_runner.tokenizer, outputs)
    return MultiGPUBenchmarkResult(
        mode="single",
        model_name=model_name,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        total_time_s=total_time,
        throughput_tok_s=total_tokens / total_time,
        ttft_ms=ttft_ms,
        peak_mem_gpu0_gb=_peak_mem_gb(0),
        peak_mem_gpu1_gb=_peak_mem_gb(1),
    )


# ── Replica benchmark ───────────────────────────────────────────────────────

def benchmark_replica(
    model_name: str,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    dtype: str = "float16",
    num_gpu_blocks: int = 200,
) -> MultiGPUBenchmarkResult:
    def _make_cfg(device: str) -> EngineConfig:
        return EngineConfig(
            model_name=model_name,
            device=device,
            dtype=dtype,
            max_batch_size=max(1, batch_size // 2 + batch_size % 2),  # 每卡最多 ceil(batch/2) 条
            block_size=256,
            num_gpu_blocks=num_gpu_blocks,
            num_hidden_layers=QWEN_LAYERS,
            num_kv_heads=QWEN_KV_HEADS,
            head_dim=QWEN_HEAD_DIM,
        )

    print(f"[replica] 初始化 ReplicaEngine: {model_name}")
    engine = ReplicaEngine(_make_cfg("cuda:0"), _make_cfg("cuda:1"))
    prompts = PROMPTS[:batch_size]

    print("[replica] 热身（2 条请求，4 token）...")
    _ = engine.generate(prompts[:2], max_new_tokens=4)
    _sync_all()

    # TTFT：单条请求（分到 engine[0]）
    _reset_peak_memory()
    t_ttft = time.perf_counter()
    _ = engine.generate(prompts[:1], max_new_tokens=1)
    _sync_all()
    ttft_ms = (time.perf_counter() - t_ttft) * 1000

    # 主 benchmark
    _reset_peak_memory()
    t0 = time.perf_counter()
    outputs = engine.generate(prompts, max_new_tokens=max_new_tokens)
    _sync_all()
    total_time = time.perf_counter() - t0

    total_tokens = _count_tokens(engine.engines[0].model_runner.tokenizer, outputs)
    return MultiGPUBenchmarkResult(
        mode="replica",
        model_name=model_name,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        total_time_s=total_time,
        throughput_tok_s=total_tokens / total_time,
        ttft_ms=ttft_ms,
        peak_mem_gpu0_gb=_peak_mem_gb(0),
        peak_mem_gpu1_gb=_peak_mem_gb(1),
    )


# ── PP（HF Pipeline Parallel）benchmark ─────────────────────────────────────

def benchmark_pp(
    model_name: str,
    batch_size: int = 8,
    max_new_tokens: int = 128,
    dtype: str = "float16",
) -> MultiGPUBenchmarkResult:
    config = EngineConfig(
        model_name=model_name,
        device="cuda:0",
        dtype=dtype,
    )
    print(f"[pp] 初始化 PPEngine（device_map='balanced'，Pipeline Parallel）: {model_name}")
    engine = PPEngine(config)
    prompts = PROMPTS[:batch_size]

    print("[pp] 热身（1 条请求，4 token）...")
    _ = engine.generate(prompts[:1], max_new_tokens=4)
    _sync_all()

    # pp 模式：TTFT 单独测单条请求
    _reset_peak_memory()
    t_ttft = time.perf_counter()
    _ = engine.generate(prompts[:1], max_new_tokens=1)
    _sync_all()
    ttft_ms = (time.perf_counter() - t_ttft) * 1000

    # 主 benchmark
    _reset_peak_memory()
    t0 = time.perf_counter()
    outputs = engine.generate(prompts, max_new_tokens=max_new_tokens)
    _sync_all()
    total_time = time.perf_counter() - t0

    total_tokens = _count_tokens(engine.tokenizer, outputs)
    return MultiGPUBenchmarkResult(
        mode="pp",
        model_name=model_name,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        total_time_s=total_time,
        throughput_tok_s=total_tokens / total_time,
        ttft_ms=ttft_ms,
        peak_mem_gpu0_gb=_peak_mem_gb(0),
        peak_mem_gpu1_gb=_peak_mem_gb(1),
    )


# ── 打印 ────────────────────────────────────────────────────────────────────

def print_result(result: MultiGPUBenchmarkResult) -> None:
    ttft_str = f"{result.ttft_ms:.1f} ms"
    print("\n========== mini-infer Phase 4 Multi-GPU Benchmark ==========")
    print(f"模式:           {result.mode}")
    print(f"模型:           {result.model_name}")
    print(f"batch_size:     {result.batch_size}")
    print(f"max_new_tokens: {result.max_new_tokens}")
    print(f"总耗时:         {result.total_time_s:.2f} s")
    print(f"Throughput:     {result.throughput_tok_s:.1f} tokens/s")
    print(f"TTFT（近似）:   {ttft_str}")
    print(f"Peak Mem GPU0:  {result.peak_mem_gpu0_gb:.2f} GB")
    print(f"Peak Mem GPU1:  {result.peak_mem_gpu1_gb:.2f} GB")
    if result.mode == "pp":
        print("注：pp 使用 HF model.generate()（Pipeline Parallel），无自定义 KV cache。")
        print("     TTFT 含 HF sampling loop 和 PP 层间激活传输开销，与 single/replica 口径不同。")
        print("     PP ≠ TP：PP 是不同层在不同 GPU；TP（Tensor Parallel）是同层按 head 切分 + all-reduce。")
    print("=============================================================\n")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="mini-infer Phase 4 multi-GPU benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["single", "replica", "pp", "tp2"],
        default="replica",
        help="推理模式：single（单卡）/ replica（双卡数据并行）/ pp（双卡 HF Pipeline Parallel，tp2 为旧别名）",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument(
        "--num-gpu-blocks",
        type=int,
        default=200,
        help="LLMEngine KV block 数。默认 200，配合 block_size=256 以适配当前 paged decode 路径。",
    )
    args = parser.parse_args()

    if args.mode == "single":
        result = benchmark_single(
            model_name=args.model,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype,
            num_gpu_blocks=args.num_gpu_blocks,
        )
    elif args.mode == "replica":
        result = benchmark_replica(
            model_name=args.model,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype,
            num_gpu_blocks=args.num_gpu_blocks,
        )
    else:  # pp or tp2 (alias)
        result = benchmark_pp(
            model_name=args.model,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype,
        )

    print_result(result)


if __name__ == "__main__":
    main()
