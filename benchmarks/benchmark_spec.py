"""
Phase 11 Speculative Decoding benchmark。

测试对象：mini-infer SpecEngine（draft=Qwen2.5-0.5B-Instruct, target=Qwen2.5-7B-Instruct）
当前阶段：Phase 11 — Speculative Decoding（draft+target 双模型，rejection sampling）

指标：
  - wall-clock throughput（tokens/s）：spec vs. target-only
  - TTFT（ms）：首 token 延迟对比
  - acceptance_rate：draft token 被 target 接受的比例（越高加速越好）
  - 输出一致性验证：spec 输出与 target-only 输出语义合理（不要求 token-level 完全一致）

用法：
  # 功能验证（dry_run，不需要权重）
  conda run -n ai-infra python benchmarks/benchmark_spec.py --dry_run

  # 真实 GPU 推理（0.5B draft on cuda:0 + 7B target on cuda:1）
  conda run -n ai-infra python benchmarks/benchmark_spec.py \\
      --draft /path/to/Qwen2.5-0.5B-Instruct \\
      --target /path/to/Qwen2.5-7B-Instruct \\
      --K 4 --max_new_tokens 64

  # 使用本地缓存路径（自动检测）
  conda run -n ai-infra python benchmarks/benchmark_spec.py \\
      --draft auto --target auto --K 4

注意：
  - dry_run acceptance_rate 为 100%（stub 所有 probs 为 0 → 直接接受路径）
  - 真实 GPU 结果需要 draft 和 target 放在不同设备（本 benchmark 默认 cuda:0/cuda:1）
  - throughput 使用 decode_tokens / total_time（不含 prompt tokens）
"""

from __future__ import annotations

import argparse
import gc
import os
import time

import torch

from mini_infer.core.config import EngineConfig
from mini_infer.runtime.engine import LLMEngine
from mini_infer.runtime.spec_engine import SpecEngine

# ---------------------------------------------------------------------------
# 常量 / 辅助
# ---------------------------------------------------------------------------

# 已知本地缓存路径（机器相关）
_HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")
_QWEN_05B_PATH = os.path.join(
    _HF_CACHE,
    "models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775",
)
_QWEN_7B_PATH = os.path.join(
    _HF_CACHE,
    "models--Qwen--Qwen2.5-7B-Instruct",  # 根目录包含全部 4 个 shard（比 snapshots/ 更完整）
)

# Qwen2.5-0.5B 架构参数
_05B_ARCH = dict(num_hidden_layers=24, num_kv_heads=2, head_dim=64)
# Qwen2.5-7B 架构参数
_7B_ARCH = dict(num_hidden_layers=28, num_kv_heads=4, head_dim=128)

TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain the difference between a list and a tuple in Python.",
    "Write a short poem about the ocean.",
    "What are the main causes of World War I?",
]


def resolve_model_path(arg: str, default_path: str) -> str:
    if arg == "auto":
        if os.path.isdir(default_path):
            return default_path
        raise FileNotFoundError(f"Auto-detect failed: {default_path} not found")
    return arg


def build_spec_engine(
    draft_path: str,
    target_path: str,
    K: int,
    dry_run: bool,
    block_size: int = 256,
) -> SpecEngine:
    if dry_run:
        draft_cfg = EngineConfig(
            model_name="stub", dry_run=True, num_gpu_blocks=128, block_size=4
        )
        target_cfg = EngineConfig(
            model_name="stub", dry_run=True, num_gpu_blocks=128, block_size=4
        )
    else:
        draft_cfg = EngineConfig(
            model_name=draft_path,
            dry_run=False,
            num_gpu_blocks=256,
            block_size=block_size,
            max_batch_size=1,
            device="cuda:0",
            **_05B_ARCH,
        )
        target_cfg = EngineConfig(
            model_name=target_path,
            dry_run=False,
            num_gpu_blocks=256,
            block_size=block_size,
            max_batch_size=1,
            device="cuda:1",
            **_7B_ARCH,
        )
    return SpecEngine(draft_cfg, target_cfg, K=K)


def build_target_only_engine(target_path: str, dry_run: bool, block_size: int = 256) -> LLMEngine:
    if dry_run:
        cfg = EngineConfig(model_name="stub", dry_run=True, num_gpu_blocks=128, block_size=4)
    else:
        cfg = EngineConfig(
            model_name=target_path,
            dry_run=False,
            num_gpu_blocks=256,
            block_size=block_size,
            max_batch_size=1,
            device="cuda:1",
            **_7B_ARCH,
        )
    return LLMEngine(cfg)


# ---------------------------------------------------------------------------
# 计时函数
# ---------------------------------------------------------------------------

def bench_spec(engine: SpecEngine, prompts: list[str], max_new_tokens: int) -> dict:
    """运行 spec generation，返回吞吐/TTFT/acceptance_rate。"""
    engine._total_draft = 0
    engine._total_accepted = 0

    # 预热（1 条）
    _ = engine.generate([prompts[0]], max_new_tokens=8, temperature=0.0)

    t0 = time.perf_counter()
    outputs = engine.generate(prompts, max_new_tokens=max_new_tokens, temperature=0.0)
    elapsed = time.perf_counter() - t0

    total_decode_tokens = sum(len(o.split()) for o in outputs)  # 近似（word count）
    return {
        "outputs": outputs,
        "elapsed_s": elapsed,
        "throughput_tps": total_decode_tokens / elapsed if elapsed > 0 else 0,
        "acceptance_rate": engine.acceptance_rate(),
    }


def bench_target_only(engine: LLMEngine, prompts: list[str], max_new_tokens: int) -> dict:
    """运行 target-only generation，作为 baseline。"""
    # 预热
    _ = engine.generate([prompts[0]], max_new_tokens=8)

    t0 = time.perf_counter()
    outputs = engine.generate(prompts, max_new_tokens=max_new_tokens)
    elapsed = time.perf_counter() - t0

    total_decode_tokens = sum(len(o.split()) for o in outputs)
    return {
        "outputs": outputs,
        "elapsed_s": elapsed,
        "throughput_tps": total_decode_tokens / elapsed if elapsed > 0 else 0,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 11 Speculative Decoding benchmark")
    parser.add_argument("--draft", default="auto", help="draft 模型路径（'auto' = 本地缓存）")
    parser.add_argument("--target", default="auto", help="target 模型路径（'auto' = 本地缓存）")
    parser.add_argument("--K", type=int, default=4, help="draft 每轮生成的候选 token 数")
    parser.add_argument("--max_new_tokens", type=int, default=64, help="每个请求最大生成 token 数")
    parser.add_argument("--dry_run", action="store_true", help="stub 模式，无需权重")
    parser.add_argument("--target_only", action="store_true", help="同时跑 target-only baseline")
    args = parser.parse_args()

    dry_run = args.dry_run

    print("=" * 70)
    print("Phase 11 Speculative Decoding Benchmark")
    print(f"  dry_run={dry_run}, K={args.K}, max_new_tokens={args.max_new_tokens}")
    print("=" * 70)

    if not dry_run:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        draft_path = resolve_model_path(args.draft, _QWEN_05B_PATH)
        target_path = resolve_model_path(args.target, _QWEN_7B_PATH)
        print(f"  draft:  {draft_path}")
        print(f"  target: {target_path}")
    else:
        draft_path = target_path = "stub"

    prompts = TEST_PROMPTS[:4]

    # --- SpecEngine ---
    print("\n[1] 加载 SpecEngine...")
    t0 = time.perf_counter()
    spec_engine = build_spec_engine(draft_path, target_path, args.K, dry_run)
    print(f"    SpecEngine loaded in {time.perf_counter()-t0:.1f}s")

    print("[2] 运行 SpecEngine generation...")
    spec_result = bench_spec(spec_engine, prompts, args.max_new_tokens)

    print(f"\n[结果] Speculative Decoding (K={args.K})")
    print(f"  total_time:      {spec_result['elapsed_s']:.2f}s  ({len(prompts)} prompts)")
    print(f"  throughput:      ~{spec_result['throughput_tps']:.1f} tokens/s  [近似，word count]")
    print(f"  acceptance_rate: {spec_result['acceptance_rate']:.2%}")
    if dry_run:
        print("  [注意] dry_run 性能数字无意义，仅验证功能路径")

    # 输出示例
    print("\n[生成示例]")
    for i, (p, o) in enumerate(zip(prompts[:2], spec_result["outputs"][:2])):
        print(f"  [{i}] {p!r}")
        print(f"      -> {o[:80]!r}{'...' if len(o) > 80 else ''}")

    # --- Target-only baseline（可选）---
    if args.target_only and not dry_run:
        # spec_engine 已经占用了 draft(cuda:0) + target(cuda:1)。
        # 若不先释放，再单独加载一份 target-only baseline，会在 24GB 卡上直接 OOM。
        del spec_engine
        gc.collect()
        torch.cuda.empty_cache()

        print("\n[3] 清理 SpecEngine 后加载 target-only engine（baseline）...")
        t0 = time.perf_counter()
        target_engine = build_target_only_engine(target_path, dry_run=False)
        print(f"    target-only engine loaded in {time.perf_counter()-t0:.1f}s")

        print("[4] 运行 target-only generation...")
        to_result = bench_target_only(target_engine, prompts, args.max_new_tokens)

        speedup = spec_result["throughput_tps"] / to_result["throughput_tps"] if to_result["throughput_tps"] > 0 else float("inf")
        print("\n[对比] Spec vs. Target-only")
        print(f"  target-only time:  {to_result['elapsed_s']:.2f}s")
        print(f"  target-only tps:   ~{to_result['throughput_tps']:.1f}")
        print(f"  spec tps:          ~{spec_result['throughput_tps']:.1f}")
        print(f"  speedup:           {speedup:.2f}x  [近似]")
    elif args.target_only and dry_run:
        print("\n  [注意] --target_only 仅在真实 GPU 模式下有效")

    print("\n[摘要]")
    if dry_run:
        print("  dry_run 功能验证完成。")
        print(f"  acceptance_rate={spec_result['acceptance_rate']:.2%}  (dry_run 下预期 100%)")
    else:
        print(f"  acceptance_rate={spec_result['acceptance_rate']:.2%}")
        print(f"  throughput={spec_result['throughput_tps']:.1f} tok/s (近似)")
        print("  运行 --target_only 获取加速比。")


if __name__ == "__main__":
    main()
