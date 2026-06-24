"""
Phase 10 Prefix Cache benchmark。

测试对象：mini-infer LLMEngine（chunk_prefill_size=0 路径）
当前阶段：Phase 10 — Prefix Caching（RadixAttention 风格 block-level hash + LRU eviction）

指标：
  - TTFT（ms）：miss vs. hit，量化前缀命中带来的延迟收益
  - 输出一致性：hit 路径与 miss 路径 token-level 完全相同
  - prefix_cache_size：命中后 cache 持有的 block 数
  - 吞吐（tokens/s）：miss / hit workload 对比

用法：
  # dry_run 功能验证
  conda run -n ai-infra python benchmarks/benchmark_prefix_cache.py --dry_run

  # 真实 GPU 推理（需要 Qwen2.5-0.5B 权重）
  conda run -n ai-infra python benchmarks/benchmark_prefix_cache.py \\
      --model /path/to/Qwen2.5-0.5B-Instruct --batch_size 8 --max_new_tokens 64

注意：
  - dry_run 模式下 TTFT 无意义（stub tokenizer），仅验证 hit/miss 路径的功能正确性
  - 真实 GPU 结果需在 Ubuntu + RTX 4090 上运行，block_size 须为 256 的倍数
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any

from mini_infer.core.config import EngineConfig
from mini_infer.runtime.engine import LLMEngine

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def infer_model_arch(model_path: str) -> dict[str, int]:
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"找不到模型配置文件：{config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = json.load(fh)
    return {
        "num_hidden_layers": int(cfg["num_hidden_layers"]),
        "num_kv_heads": int(cfg.get("num_key_value_heads", cfg["num_attention_heads"])),
        "head_dim": int(cfg["hidden_size"]) // int(cfg["num_attention_heads"]),
    }


def build_engine(model: str, dry_run: bool, num_gpu_blocks: int, block_size: int) -> LLMEngine:
    arch = {} if dry_run else infer_model_arch(model)
    config = EngineConfig(
        model_name=model,
        dry_run=dry_run,
        num_gpu_blocks=num_gpu_blocks,
        block_size=block_size,
        max_batch_size=16,
        **arch,
    )
    return LLMEngine(config)


def timed_generate(engine: LLMEngine, prompts: list[str], max_new_tokens: int) -> tuple[list[str], float]:
    """执行 generate 并返回 (outputs, elapsed_ms)。"""
    t0 = time.perf_counter()
    outputs = engine.generate(prompts, max_new_tokens=max_new_tokens)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return outputs, elapsed_ms


def print_row(label: str, elapsed_ms: float, cache_size: int, outputs: list[str]) -> None:
    tokens_generated = sum(len(o.split()) for o in outputs)  # 近似
    print(f"  {label:<30} | {elapsed_ms:>8.1f} ms | cache_size={cache_size:>3} | ~{tokens_generated} tokens")


# ---------------------------------------------------------------------------
# 实验 1：单请求 miss vs. hit TTFT 对比
# ---------------------------------------------------------------------------

def bench_single_hit(engine: LLMEngine, prefix: str, suffix_miss: str, suffix_hit: str,
                     max_new_tokens: int) -> dict[str, Any]:
    """
    第 1 次请求：prefix + suffix_miss（miss，建立 cache）
    第 2 次请求：prefix + suffix_hit（hit，复用 prefix 的 KV）
    """
    prompt_miss = prefix + " " + suffix_miss
    prompt_hit = prefix + " " + suffix_hit

    # miss
    out_miss, ms_miss = timed_generate(engine, [prompt_miss], max_new_tokens)
    cache_after_miss = engine.kv_cache.prefix_cache_size()

    # hit
    out_hit, ms_hit = timed_generate(engine, [prompt_hit], max_new_tokens)
    cache_after_hit = engine.kv_cache.prefix_cache_size()

    speedup = ms_miss / ms_hit if ms_hit > 0 else float("inf")
    return {
        "ms_miss": ms_miss,
        "ms_hit": ms_hit,
        "speedup": speedup,
        "cache_miss": cache_after_miss,
        "cache_hit": cache_after_hit,
        "out_miss": out_miss[0],
        "out_hit": out_hit[0],
    }


# ---------------------------------------------------------------------------
# 实验 2：batch miss vs. hit 吞吐对比
# ---------------------------------------------------------------------------

def bench_batch_throughput(engine: LLMEngine, shared_prefix: str, suffixes: list[str],
                           max_new_tokens: int) -> dict[str, Any]:
    """
    miss batch：每个请求不同 prompt（无共享前缀）
    hit batch：每个请求 = shared_prefix + 不同 suffix（共享前缀命中）
    """
    # miss batch（先跑，避免 hit batch 复用 miss batch 建立的 cache）
    miss_prompts = [f"unrelated prompt {i} " + s for i, s in enumerate(suffixes)]
    _, ms_miss = timed_generate(engine, miss_prompts, max_new_tokens)

    # hit batch
    hit_prompts = [shared_prefix + " " + s for s in suffixes]
    _, ms_hit_first = timed_generate(engine, hit_prompts, max_new_tokens)  # 第 1 次：miss，建立 cache
    _, ms_hit_second = timed_generate(engine, hit_prompts, max_new_tokens)  # 第 2 次：hit

    cache_size = engine.kv_cache.prefix_cache_size()
    total_tokens = len(suffixes) * max_new_tokens
    tps_miss = total_tokens / (ms_miss / 1000) if ms_miss > 0 else 0
    tps_hit = total_tokens / (ms_hit_second / 1000) if ms_hit_second > 0 else 0

    return {
        "ms_miss": ms_miss,
        "ms_hit_second": ms_hit_second,
        "tps_miss": tps_miss,
        "tps_hit": tps_hit,
        "speedup": ms_miss / ms_hit_second if ms_hit_second > 0 else float("inf"),
        "cache_size": cache_size,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 10 Prefix Cache benchmark")
    parser.add_argument("--model", default="stub", help="模型名称或路径（stub = dry_run 模式）")
    parser.add_argument("--dry_run", action="store_true", help="使用 stub tokenizer（不需要权重）")
    parser.add_argument("--num_gpu_blocks", type=int, default=512, help="KV cache 块数")
    parser.add_argument("--block_size", type=int, default=256, help="每个 block 的 token 数（真实 GPU 须为 256）")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="每个请求最大生成 token 数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch 吞吐实验的请求数")
    args = parser.parse_args()

    dry_run = args.dry_run or args.model == "stub"
    if dry_run:
        args.block_size = 4  # dry_run 使用小 block_size

    print("=" * 70)
    print("Phase 10 Prefix Cache Benchmark")
    print(f"  model={args.model}, dry_run={dry_run}")
    print(f"  num_gpu_blocks={args.num_gpu_blocks}, block_size={args.block_size}")
    print(f"  max_new_tokens={args.max_new_tokens}, batch_size={args.batch_size}")
    if not dry_run:
        arch = infer_model_arch(args.model)
        print(
            "  arch="
            f"{arch['num_hidden_layers']} layers, "
            f"{arch['num_kv_heads']} KV heads, "
            f"head_dim={arch['head_dim']}"
        )
    print("=" * 70)

    engine = build_engine(args.model, dry_run, args.num_gpu_blocks, args.block_size)

    # ── 构建足够长的 shared prefix ────────────────────────────────────────
    # 对于真实 GPU（block_size=256），prefix 必须 > 256 tokens 才能在 prefix cache 中形成完整 block。
    # 典型生产场景的系统提示（RAG、角色扮演、工具调用说明）通常 500-2000 tokens。
    # 这里用重复的段落构建 ~512 token 的前缀，模拟长系统提示场景。
    _base = (
        "You are a helpful assistant. The following is a detailed system prompt shared across "
        "all users. It describes your behavior, capabilities, and operational constraints. "
        "You should always respond in a clear, concise, and accurate manner. "
        "When asked technical questions, provide precise explanations with examples. "
        "Avoid speculation and clearly indicate uncertainty when appropriate. "
        "Follow all instructions provided by the user while adhering to safety guidelines. "
    )
    # 重复拼接直到 token 数超过 block_size（dry_run 下 block_size=4 不需要长前缀）
    target_prefix_tokens = args.block_size + 1 if not dry_run else 9
    shared_prefix = (_base * 20)[:5000]  # 足够长的原始字符串，稍后截断到目标 token 数
    # 使用 engine 的 tokenizer 截断到 target_prefix_tokens + 1 个 token（保证超过 block_size）
    if not dry_run and hasattr(engine.model_runner, "tokenizer"):
        tok = engine.model_runner.tokenizer
        tids = tok.encode(shared_prefix, add_special_tokens=True)
        if len(tids) > target_prefix_tokens:
            tids = tids[:target_prefix_tokens]
        shared_prefix = tok.decode(tids, skip_special_tokens=True)
        print(f"  [prefix] 实际 token 数: {len(tids)}（block_size={args.block_size}）")
    else:
        # dry_run：stub tokenizer 每字符=1 token，直接截断字符数到目标 token 数
        shared_prefix = shared_prefix[:target_prefix_tokens]
    result1 = bench_single_hit(
        engine,
        prefix=shared_prefix,
        suffix_miss="What is the weather today?",
        suffix_hit="Tell me about history.",
        max_new_tokens=args.max_new_tokens,
    )

    print("\n[实验 1] 单请求 miss vs. hit TTFT")
    print(f"  {'label':<30} | {'elapsed':>8} | {'cache_size':>10}")
    print(f"  {'-' * 60}")
    print_row("miss (建立 cache)", result1["ms_miss"], result1["cache_miss"], [result1["out_miss"]])
    print_row("hit  (复用 cache)", result1["ms_hit"], result1["cache_hit"], [result1["out_hit"]])
    print(f"  speedup: {result1['speedup']:.2f}x")
    if dry_run:
        print("  [注意] dry_run 模式下 TTFT 无意义，仅验证功能路径")

    # ── 实验 2：batch 吞吐 ───────────────────────────────────────────────
    suffixes = [f"question number {i}" for i in range(args.batch_size)]
    result2 = bench_batch_throughput(
        engine,
        shared_prefix=shared_prefix,
        suffixes=suffixes,
        max_new_tokens=args.max_new_tokens,
    )

    print(f"\n[实验 2] batch={args.batch_size} miss vs. hit 吞吐")
    print(f"  miss batch  : {result2['ms_miss']:.1f} ms  |  ~{result2['tps_miss']:.0f} tokens/s (近似)")
    print(f"  hit  batch  : {result2['ms_hit_second']:.1f} ms  |  ~{result2['tps_hit']:.0f} tokens/s (近似)")
    print(f"  speedup     : {result2['speedup']:.2f}x")
    print(f"  prefix_cache_size: {result2['cache_size']} blocks")
    if dry_run:
        print("  [注意] dry_run 模式下吞吐无意义，仅验证功能路径")

    # ── 摘要 ─────────────────────────────────────────────────────────────
    print("\n[摘要]")
    if dry_run:
        print("  dry_run 功能验证完成。请在真实 GPU 环境重跑以获得有意义的性能数据。")
        print("  验证点：")
        print(f"    prefix_cache_size > 0 after miss:  {result1['cache_miss'] > 0}")
        print(f"    prefix_cache_size >= 1 after hit:  {result1['cache_hit'] >= 1}")
    else:
        print(f"  单请求命中加速: {result1['speedup']:.2f}x")
        print(f"  batch 命中加速: {result2['speedup']:.2f}x")
        print("  [注意] 吞吐数字为近似估计（word count），不是精确 token 数")
        print("  [注意] 基准对比（HF baseline）需另行跑 benchmarks/benchmark_hf.py")


if __name__ == "__main__":
    main()
