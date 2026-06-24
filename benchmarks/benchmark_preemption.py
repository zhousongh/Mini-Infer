"""
benchmarks/benchmark_preemption.py — Phase 7 Preemption + Priority Scheduling benchmark。

测试对象：mini-infer LLMEngine（GPU 实机 + Qwen2.5-7B-Instruct）

测量内容：
  1. swap_out / swap_in 真实延迟（ms，含 GPU→CPU tensor 拷贝，不含模型加载）
  2. 有无抢占时的端到端吞吐对比（tok/s，基于 benchmark_mini.py 精确口径）
  3. 调度器纯逻辑延迟（µs，dry_run）

运行方式：
  export MODEL_PATH=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
  export HF_HUB_OFFLINE=1
  conda run -n ai-infra python benchmarks/benchmark_preemption.py
  # 只跑调度器 + swap 延迟（无需等模型加载）：加 --dry-only

Workload（GPU 测试）：
  - 模型：Qwen2.5-7B-Instruct（本地路径由 MODEL_PATH 覆盖）
  - swap latency：prompt_len=32 token，block_size=16（不经过 flash_attn）
  - 吞吐回归：prompt_len=16~64 token，max_new_tokens=32，block_size=256（flash_attn 路径）
  - batch_size：4
"""

import math
import os
import sys
import time

import torch

# 允许从项目根目录运行
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mini_infer import EngineConfig, LLMEngine

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
    ),
)

# ── 参数 ────────────────────────────────────────────────────────────────────
BLOCK_SIZE = 16        # swap latency 测试：任意 block_size 均可（不调 flash_attn）
GPU_BLOCK_SIZE = 256   # GPU 吞吐测试：flash_attn_with_kvcache 需要 256 对齐
NUM_GPU_BLOCKS = 64    # swap latency 测试用
GPU_NUM_BLOCKS = 512   # 吞吐测试用（block_size=256，512 块 = 131072 tokens）
MAX_BATCH_SIZE = 4
MAX_NEW_TOKENS = 32

# 用于 swap 延迟测量的 prompt（确保需要 ceil((len+max_out)/block_size) 块）
SWAP_PROMPTS = ["The quick brown fox"] * 2

# 吞吐测试 prompt 集
THROUGHPUT_PROMPTS = [
    "Tell me about artificial intelligence.",
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a short poem about the ocean.",
    "How does photosynthesis work?",
    "What are the benefits of exercise?",
    "Describe the solar system.",
    "What is machine learning?",
]

WARMUP_PROMPTS = ["Hello world"] * 2


def make_engine(num_gpu_blocks: int = NUM_GPU_BLOCKS) -> LLMEngine:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"模型权重未找到：{MODEL_PATH}\n"
            "请设置环境变量 MODEL_PATH 指向正确路径，或使用 dry_run=True 模式。"
        )
    config = EngineConfig(
        model_name=MODEL_PATH,
        dry_run=False,
        block_size=GPU_BLOCK_SIZE,
        num_gpu_blocks=num_gpu_blocks,
        max_batch_size=MAX_BATCH_SIZE,
    )
    return LLMEngine(config)


def make_dry_run_engine(
    num_gpu_blocks: int = NUM_GPU_BLOCKS,
    max_batch_size: int = MAX_BATCH_SIZE,
) -> LLMEngine:
    config = EngineConfig(
        model_name="stub",
        dry_run=True,
        block_size=BLOCK_SIZE,
        num_gpu_blocks=num_gpu_blocks,
        max_batch_size=max_batch_size,
    )
    return LLMEngine(config)


# ── 测试 1：dry_run 下调度延迟 ───────────────────────────────────────────────

def bench_scheduler_latency(n_trials: int = 1000) -> None:
    """在 dry_run 模式下测量 swap_out / swap_in 调度路径的延迟（不含模型）。"""
    print("\n=== Scheduler latency（dry_run，不含模型 forward）===")

    from mini_infer.cache.kv_cache import KVCacheManager
    from mini_infer.core.request import Request, RequestState, SamplingParams

    config = EngineConfig(
        model_name="stub",
        dry_run=True,
        block_size=BLOCK_SIZE,
        num_gpu_blocks=64,
        max_batch_size=8,
    )
    mgr = KVCacheManager(config)

    def make_state(i: int) -> RequestState:
        return RequestState(
            request=Request(
                request_id=f"req-{i}",
                prompt="x" * 32,
                sampling_params=SamplingParams(max_new_tokens=MAX_NEW_TOKENS),
            ),
            prompt_token_ids=[1] * 32,
        )

    # 预热
    for i in range(10):
        s = make_state(i)
        mgr.init_request(s)
        mgr.swap_out(s)
        mgr.swap_in(s)
        mgr.free_request(s)

    # 正式测量
    swap_out_times = []
    swap_in_times = []

    for i in range(n_trials):
        s = make_state(i + 100)
        mgr.init_request(s)

        t0 = time.perf_counter()
        mgr.swap_out(s)
        t1 = time.perf_counter()
        mgr.swap_in(s)
        t2 = time.perf_counter()

        swap_out_times.append((t1 - t0) * 1e6)
        swap_in_times.append((t2 - t1) * 1e6)

        mgr.free_request(s)

    def stats(xs: list[float]) -> str:
        import statistics
        return (
            f"mean={statistics.mean(xs):.2f}µs  "
            f"median={statistics.median(xs):.2f}µs  "
            f"p99={sorted(xs)[int(len(xs) * 0.99)]:.2f}µs"
        )

    print(f"  swap_out ({n_trials} trials): {stats(swap_out_times)}")
    print(f"  swap_in  ({n_trials} trials): {stats(swap_in_times)}")
    print("  注：dry_run 模式，只含块分配/释放逻辑，无 tensor 拷贝。")


# ── 测试 2：dry_run 端到端，有/无抢占吞吐对比 ────────────────────────────────

def bench_e2e_dry_run() -> None:
    """dry_run 端到端调度延迟：无抢占 vs 有抢占。"""
    print("\n=== E2E dry_run 端到端对比 ===")

    prompts = ["ab cd ef gh"] * 4  # 每个 ~8 tokens

    # 无抢占（充足 KV 块）
    engine_no_preempt = make_dry_run_engine(num_gpu_blocks=32, max_batch_size=4)
    t0 = time.perf_counter()
    out_no = engine_no_preempt.generate(prompts, max_new_tokens=8)
    t1 = time.perf_counter()
    elapsed_no = (t1 - t0) * 1000

    # 有抢占（KV 块受限，低优先级被换出）
    engine_preempt = make_dry_run_engine(num_gpu_blocks=2, max_batch_size=4)
    priorities = [5, 5, 0, 0]  # 前两个低优先级，后两个高优先级
    t0 = time.perf_counter()
    out_yes = engine_preempt.generate(prompts, max_new_tokens=4, priorities=priorities)
    t1 = time.perf_counter()
    elapsed_yes = (t1 - t0) * 1000

    print(f"  无抢占（充足块）：{elapsed_no:.1f}ms，输出 {len(out_no)} 条")
    print(f"  有抢占（限制块）：{elapsed_yes:.1f}ms，输出 {len(out_yes)} 条")
    print("  注：dry_run 模式，纯调度逻辑，不含模型推理。")


# ── 测试 3：真实 GPU 吞吐回归（需要模型权重，block_size=256）───────────────

def _make_gpu_engine(num_gpu_blocks: int = GPU_NUM_BLOCKS) -> LLMEngine:
    """创建 GPU 引擎（block_size=256，适配 flash_attn_with_kvcache）。"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"模型权重未找到：{MODEL_PATH}\n"
            "请设置 MODEL_PATH 环境变量，并确认 HF_HUB_OFFLINE=1"
        )
    config = EngineConfig(
        model_name=MODEL_PATH,
        dry_run=False,
        block_size=GPU_BLOCK_SIZE,   # 256，flash_attn 兼容
        num_gpu_blocks=num_gpu_blocks,
        max_batch_size=MAX_BATCH_SIZE,
    )
    return LLMEngine(config)


def bench_gpu_throughput() -> None:
    """
    GPU 实机：Phase 7 吞吐回归检查。

    使用 benchmark_mini.py 相同的精确 token 计数口径（tokenizer.encode 长度）。
    Phase 6 baseline：batch=8 时约 406 tok/s（= 100% HF）。
    Phase 7 引入调度层变更，decode 路径不变，期望吞吐无回归。
    """
    print("\n=== GPU 吞吐回归检查（block_size=256，flash_attn 路径）===")

    try:
        engine = _make_gpu_engine()
    except FileNotFoundError as e:
        print(f"  跳过（{e}）")
        return

    # 预热
    print("  加载模型 + 预热中...")
    engine.generate(WARMUP_PROMPTS, max_new_tokens=8)
    torch.cuda.synchronize()

    # ── 测量 1：无抢占，batch=4，精确 token 计数 ─────────────────────────
    print("  [1/2] 测量正常路径（无抢占）...")
    t0 = time.perf_counter()
    outputs_base = engine.generate(THROUGHPUT_PROMPTS[:4], max_new_tokens=MAX_NEW_TOKENS)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed_base = t1 - t0

    # 精确 token 计数（使用 tokenizer）
    total_tokens_base = sum(
        len(engine.model_runner.tokenizer.encode(o, add_special_tokens=False))
        for o in outputs_base
    )
    throughput_base = total_tokens_base / elapsed_base
    print(f"  正常路径：{elapsed_base:.2f}s，{total_tokens_base} tokens，"
          f"{throughput_base:.1f} tok/s  [batch=4, max_new_tokens={MAX_NEW_TOKENS}]")

    # ── 测量 2：有优先级差异，batch=4，不触发真实抢占（充足 KV 块）──────
    print("  [2/2] 测量 priority 调度路径（充足块，不触发换出）...")
    # 所有请求使用不同优先级，但块充足，不需要换出
    priorities_mixed = [3, 1, 2, 0]
    t0 = time.perf_counter()
    outputs_prio = engine.generate(
        THROUGHPUT_PROMPTS[4:8], max_new_tokens=MAX_NEW_TOKENS, priorities=priorities_mixed
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed_prio = t1 - t0
    total_tokens_prio = sum(
        len(engine.model_runner.tokenizer.encode(o, add_special_tokens=False))
        for o in outputs_prio
    )
    throughput_prio = total_tokens_prio / elapsed_prio
    print(f"  priority 路径：{elapsed_prio:.2f}s，{total_tokens_prio} tokens，"
          f"{throughput_prio:.1f} tok/s  [batch=4, max_new_tokens={MAX_NEW_TOKENS}]")

    delta = (throughput_prio - throughput_base) / throughput_base * 100
    print(f"  与 Phase 6 baseline 对比：Phase 6 batch=8 约 406 tok/s；"
          f"本次 batch=4 {throughput_base:.1f} tok/s（不同 batch，仅供参考）")
    print(f"  priority 路径相对正常路径：{delta:+.1f}%（期望接近 0%）")
    print(f"  模型：{os.path.basename(MODEL_PATH)}, block_size={GPU_BLOCK_SIZE}")


# ── 测试 4：GPU swap 真实延迟（直接用 KVCacheManager，无需模型加载）───────

def bench_gpu_swap_latency() -> None:
    """
    GPU 实机：测量单次 swap_out + swap_in 的真实延迟（含 GPU→CPU tensor 拷贝）。

    直接创建 KVCacheManager（dry_run=False），不加载模型权重。
    block_size=16 对此测试无约束（不经过 flash_attn），且块粒度更细，适合延迟测量。
    测量维度：不同 seq_len（32 / 256 / 512 token）下的拷贝延迟。
    """
    print("\n=== GPU Swap 真实延迟（含 tensor 拷贝，不含模型加载）===")

    from mini_infer.cache.kv_cache import KVCacheManager
    from mini_infer.core.request import Request, RequestState, SamplingParams

    # 直接构建 KVCacheManager，用 7B 的实际参数（num_layers=28, num_kv_heads=4, head_dim=128）
    config = EngineConfig(
        model_name="stub",   # 不加载模型权重
        dry_run=False,       # 分配真实 GPU tensor
        block_size=BLOCK_SIZE,
        num_gpu_blocks=256,
        max_batch_size=MAX_BATCH_SIZE,
        # 7B 参数（与实际模型一致）
        num_hidden_layers=28,
        num_kv_heads=4,
        head_dim=128,
        dtype="float16",
        device="cuda:0",
    )
    mgr = KVCacheManager(config)
    torch.cuda.synchronize()  # 等待 GPU tensor 分配完成

    n_trials = 50

    for prompt_len in [32, 256, 512]:
        num_blocks = max(1, math.ceil(prompt_len / BLOCK_SIZE))
        if num_blocks > 256:
            print(f"  seq_len={prompt_len}: 跳过（需要 {num_blocks} 块，超过预分配 256 块）")
            continue

        swap_out_ms = []
        swap_in_ms = []

        for i in range(n_trials):
            state = RequestState(
                request=Request(
                    request_id=f"bench-{prompt_len}-{i}",
                    prompt="x" * prompt_len,
                    sampling_params=SamplingParams(max_new_tokens=MAX_NEW_TOKENS),
                ),
                prompt_token_ids=[1] * prompt_len,
            )
            mgr.init_request(state)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            mgr.swap_out(state)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            mgr.swap_in(state)
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            swap_out_ms.append((t1 - t0) * 1000)
            swap_in_ms.append((t2 - t1) * 1000)
            mgr.free_request(state)

        import statistics

        # 计算拷贝数据量
        kv_mb = prompt_len * 4 * 128 * 28 * 2 * 2 / 1e6  # K+V, fp16
        mean_out = statistics.mean(swap_out_ms)
        mean_in = statistics.mean(swap_in_ms)
        bw_out = kv_mb / (mean_out / 1000) if mean_out > 0 else 0  # MB/s
        bw_in = kv_mb / (mean_in / 1000) if mean_in > 0 else 0

        print(
            f"  seq_len={prompt_len:4d} ({kv_mb:.1f} MB KV): "
            f"swap_out={mean_out:.2f}ms ({bw_out:.0f} MB/s)  "
            f"swap_in={mean_in:.2f}ms ({bw_in:.0f} MB/s)  "
            f"[n={n_trials}]"
        )

    print(f"  block_size={BLOCK_SIZE}, num_layers=28, num_kv_heads=4, head_dim=128, dtype=fp16")


# ── main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Phase 7 Preemption Benchmark")
    parser.add_argument(
        "--dry-only",
        action="store_true",
        help="只跑 dry_run 测试（无需 GPU/模型权重）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Phase 7 Preemption + Priority Scheduling Benchmark")
    print("=" * 60)

    bench_scheduler_latency()
    bench_e2e_dry_run()

    if not args.dry_only:
        bench_gpu_swap_latency()
        bench_gpu_throughput()
    else:
        print("\n（--dry-only 模式：跳过 GPU 测试）")

    print("\n✓ benchmark 完成")
