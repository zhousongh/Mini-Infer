"""
Phase 15 Benchmark：PD 解耦（Disaggregated Prefill/Decode）

Section 1：理论 KV 传输大小（CPU，无 GPU）
Section 2：真实 GPU 端到端验证（Qwen2.5-1.5B，生成结果对比）
Section 3：TTFT 分解（prefill_time / transfer_time / decode_time）+ 吞吐对比

用法：
    python benchmarks/benchmark_pd_disagg.py --section 1
    python benchmarks/benchmark_pd_disagg.py --section 2
    python benchmarks/benchmark_pd_disagg.py --section 3
    python benchmarks/benchmark_pd_disagg.py  # 全部运行
"""

from __future__ import annotations

import argparse
import time

import torch

# ──────────────────────────────────────────────────────────────────────────────
# Section 1：理论 KV 传输大小
# ──────────────────────────────────────────────────────────────────────────────

def section1_theoretical():
    print("=" * 60)
    print("Section 1：理论 KV 传输大小")
    print("=" * 60)

    # GQA 模型：存储分开的 K 和 V，× 2
    # MLA 模型：只存储一个压缩向量（compressed_kv + k_rope），不分 K/V，× 1
    configs = [
        ("Qwen2.5-1.5B", 28, 2, 128, 2, "GQA"),
        ("Qwen2.5-7B",   28, 4, 128, 2, "GQA"),
        ("DeepSeek-V2-Lite MLA", 27, 1, 576, 1, "MLA latent+rope（单向量）"),
    ]

    seq_lens = [128, 512, 1024]
    bytes_per_elem = 2  # fp16

    print(f"\n{'模型':<32} {'seq_len':<10} {'KV传输大小':>15} {'说明'}")
    print("-" * 78)
    for name, num_layers, num_kv_heads, head_dim, kv_factor, arch in configs:
        for seq_len in seq_lens:
            # kv_factor=2 表示 K+V 两路；MLA latent 只有一路（×1）
            kv_bytes = num_layers * seq_len * num_kv_heads * head_dim * kv_factor * bytes_per_elem
            kv_mb = kv_bytes / 1024 / 1024
            print(f"{name:<32} {seq_len:<10} {kv_mb:>12.2f} MB  {arch}")
        print()

    print("结论：同机共享内存传输，即使 7B seq=1024 也只需 ~56 MB，延迟预期 < 5ms。")


# ──────────────────────────────────────────────────────────────────────────────
# Section 2：真实 GPU 端到端验证
# ──────────────────────────────────────────────────────────────────────────────

def section2_gpu_equivalence():
    print("=" * 60)
    print("Section 2：真实 GPU 端到端验证（Qwen2.5-1.5B）")
    print("=" * 60)

    import os
    os.environ["HF_HUB_OFFLINE"] = "1"

    from mini_infer.core.config import EngineConfig
    from mini_infer.runtime.engine import LLMEngine
    from mini_infer.runtime.pd_engine import PDEngine

    MODEL = "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    MODEL = os.path.expanduser(MODEL)

    config = EngineConfig(
        model_name=MODEL,
        device="cuda:0",
        dtype="float16",
        num_hidden_layers=28,
        num_kv_heads=2,
        head_dim=128,
        max_batch_size=1,
        block_size=256,
        num_gpu_blocks=50,
    )

    prompts = ["What is the capital of France?"]
    max_new_tokens = 32

    print("\n[LLMEngine] 生成中...")
    unified_engine = LLMEngine(config)
    unified_results = unified_engine.generate(prompts, max_new_tokens=max_new_tokens)
    print(f"  输出：{unified_results[0][:80]!r}")
    del unified_engine
    torch.cuda.empty_cache()

    print("\n[PDEngine] 生成中...")
    # PDEngine 使用不同 device 避免显存冲突（两进程各加载一份模型）
    pd_config = EngineConfig(
        model_name=MODEL,
        device="cuda:0",
        dtype="float16",
        num_hidden_layers=28,
        num_kv_heads=2,
        head_dim=128,
        max_batch_size=1,
        block_size=256,
        num_gpu_blocks=50,
    )
    with PDEngine(pd_config) as pd_engine:
        pd_results = pd_engine.generate(prompts, max_new_tokens=max_new_tokens, timeout=120.0)
    print(f"  输出：{pd_results[0][:80]!r}")

    print("\n[对比]")
    match = unified_results[0].strip() == pd_results[0].strip()
    print(f"  结果一致：{'✓' if match else '✗（注意：两进程各自加载模型，greedy 应一致）'}")


# ──────────────────────────────────────────────────────────────────────────────
# Section 3：TTFT 分解 + 吞吐对比
# ──────────────────────────────────────────────────────────────────────────────

def section3_timing():
    print("=" * 60)
    print("Section 3：TTFT 分解（prefill / transfer / decode）")
    print("=" * 60)

    import os
    os.environ["HF_HUB_OFFLINE"] = "1"

    from mini_infer.core.config import EngineConfig
    from mini_infer.runtime.engine import LLMEngine
    from mini_infer.runtime.pd_engine import PDEngine

    MODEL = "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
    MODEL = os.path.expanduser(MODEL)

    config = EngineConfig(
        model_name=MODEL,
        device="cuda:0",
        dtype="float16",
        num_hidden_layers=28,
        num_kv_heads=2,
        head_dim=128,
        max_batch_size=1,
        block_size=256,
        num_gpu_blocks=50,
    )

    prompts = [
        "Explain the concept of attention mechanism in transformers.",
        "What are the main differences between Python 2 and Python 3?",
        "Describe the architecture of a typical neural network.",
    ]
    max_new_tokens = 64
    n_warmup = 1

    # ── Unified LLMEngine baseline ──
    print("\n[Unified LLMEngine] 预热 + 计时...")
    unified_engine = LLMEngine(config)
    for _ in range(n_warmup):
        unified_engine.generate([prompts[0]], max_new_tokens=max_new_tokens)

    unified_times = []
    for p in prompts:
        t0 = time.perf_counter()
        unified_engine.generate([p], max_new_tokens=max_new_tokens)
        unified_times.append(time.perf_counter() - t0)

    del unified_engine
    torch.cuda.empty_cache()

    avg_unified = sum(unified_times) / len(unified_times)
    print(f"  平均端到端时间：{avg_unified*1000:.1f} ms")

    # ── PDEngine ──
    print("\n[PDEngine] 预热 + 计时...")
    with PDEngine(config) as pd_engine:
        # 预热
        for _ in range(n_warmup):
            pd_engine.generate([prompts[0]], max_new_tokens=max_new_tokens, timeout=120.0)

        pd_timings = []
        for p in prompts:
            t0 = time.perf_counter()
            timing_results = pd_engine.generate_with_timing(
                [p], max_new_tokens=max_new_tokens, timeout=120.0
            )
            total_time = time.perf_counter() - t0
            r = timing_results[0]
            pd_timings.append({
                "total": total_time,
                "prefill_time": r["prefill_time"],
                "decode_time": r["decode_time"],
            })

    avg_pd_total = sum(t["total"] for t in pd_timings) / len(pd_timings)
    avg_pd_prefill = sum(t["prefill_time"] for t in pd_timings) / len(pd_timings)
    avg_pd_decode = sum(t["decode_time"] for t in pd_timings) / len(pd_timings)
    # transfer_time ≈ total - prefill - decode（含 IPC queue 开销）
    avg_pd_transfer = avg_pd_total - avg_pd_prefill - avg_pd_decode

    print(f"\n{'指标':<30} {'Unified':>15} {'PDEngine':>15}")
    print("-" * 62)
    print(f"{'平均端到端时间 (ms)':<30} {avg_unified*1000:>15.1f} {avg_pd_total*1000:>15.1f}")
    print(f"{'  prefill 时间 (ms)':<30} {'N/A':>15} {avg_pd_prefill*1000:>15.1f}")
    print(f"  {'transfer 时间 (近似, ms)':<28} {'N/A':>15} {avg_pd_transfer*1000:>15.1f}")
    print(f"{'  decode 时间 (ms)':<30} {'N/A':>15} {avg_pd_decode*1000:>15.1f}")
    print(f"{'开销比':<30} {'1.00×':>15} {avg_pd_total/avg_unified:>14.2f}×")

    print("\n注：PDEngine 两进程各加载一份模型，显存占用 2×；")
    print("    端到端时间包含进程间 Queue 传输开销（pickle + IPC）。")
    print("    生产系统用 RDMA/共享内存可将传输延迟降至 < 1ms。")


# ──────────────────────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", type=int, choices=[1, 2, 3], default=0,
                        help="运行指定 section（默认全部）")
    args = parser.parse_args()

    if args.section == 0 or args.section == 1:
        section1_theoretical()
    if args.section == 0 or args.section == 2:
        section2_gpu_equivalence()
    if args.section == 0 or args.section == 3:
        section3_timing()
