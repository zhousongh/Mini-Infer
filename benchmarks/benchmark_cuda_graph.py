"""
Phase 12 CUDA Graph Benchmark。

对比 decode_batch 在 eager 模式和 CUDA Graph 模式下的延迟与吞吐。

测量对象：
  - mini-infer LLMEngine（eager，use_cuda_graph=False）
  - mini-infer LLMEngine（graph，use_cuda_graph=True）
  - 对比维度：decode step 延迟（ms/step）、throughput（tok/s）

Workload：
  - 模型：Qwen2.5-1.5B-Instruct（默认）或指定 --model
  - Batch size：1 / 2 / 4 / 8（逐个测量）
  - Decode steps：50（warmup 10 步）
  - temperature=0（greedy），固定 prompt

运行方式（在项目根目录，ai-infra 环境）：
  export HF_HUB_OFFLINE=1
  python benchmarks/benchmark_cuda_graph.py --model <MODEL_PATH>
"""

import argparse
import time

import torch

from mini_infer.core.config import EngineConfig
from mini_infer.runtime.engine import LLMEngine

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming the way we work.",
    "In the beginning was the Word, and the Word was with God.",
    "It was the best of times, it was the worst of times.",
    "To be, or not to be, that is the question.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "Call me Ishmael.",
    "It is a truth universally acknowledged that a single man in possession of a good fortune must be in want of a wife.",
]


def _make_engine(model_path: str, device: str, use_cuda_graph: bool, num_kv_heads: int, head_dim: int, num_layers: int) -> LLMEngine:
    cfg = EngineConfig(
        model_name=model_path,
        device=device,
        dtype="float16",
        max_batch_size=8,
        max_model_len=1024,
        block_size=256,
        num_gpu_blocks=200,
        num_hidden_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        use_cuda_graph=use_cuda_graph,
    )
    return LLMEngine(cfg)


def _benchmark_one(engine: LLMEngine, batch_size: int, decode_steps: int, warmup_steps: int) -> dict:
    prompts = PROMPTS[:batch_size]

    # Warmup
    engine.generate(prompts, max_new_tokens=warmup_steps, temperature=0.0)

    # Timed run
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    engine.generate(prompts, max_new_tokens=decode_steps, temperature=0.0)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    elapsed = t1 - t0
    total_tokens = batch_size * decode_steps
    throughput = total_tokens / elapsed
    # latency_per_step_ms：elapsed / decode_steps，包含 prefill 耗时（近似值）。
    # prefill 走 eager 路径，两种模式下相同，不影响 eager vs graph 的相对比较。
    latency_per_step_ms = elapsed / decode_steps * 1000

    return {
        "batch_size": batch_size,
        "total_tokens": total_tokens,
        "elapsed_s": round(elapsed, 3),
        "throughput_tok_s": round(throughput, 1),
        "latency_ms_per_step": round(latency_per_step_ms, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 12 CUDA Graph Benchmark")
    parser.add_argument("--model", required=True, help="模型路径")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decode-steps", type=int, default=50)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--num-kv-heads", type=int, default=2, help="1.5B=2, 7B=4")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=28)
    args = parser.parse_args()

    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    print(f"Decode steps: {args.decode_steps}, warmup: {args.warmup_steps}")
    print()

    print("=" * 60)
    print(f"{'bs':>4}  {'mode':>8}  {'lat(ms/step)':>14}  {'tok/s':>8}")
    print("-" * 60)

    # 为两种模式各建一个 engine（避免互相干扰）
    for use_graph in [False, True]:
        mode_name = "graph" if use_graph else "eager"
        engine = _make_engine(
            args.model, args.device, use_graph,
            args.num_kv_heads, args.head_dim, args.num_layers,
        )
        for bs in args.batch_sizes:
            result = _benchmark_one(engine, bs, args.decode_steps, args.warmup_steps)
            print(
                f"{bs:>4}  {mode_name:>8}  "
                f"{result['latency_ms_per_step']:>14.2f}  "
                f"{result['throughput_tok_s']:>8.1f}"
            )
        del engine
        torch.cuda.empty_cache()
        print()

    print("=" * 60)
    print("注：throughput = batch_size × decode_steps / elapsed（近似，非精确 token count）")


if __name__ == "__main__":
    main()
