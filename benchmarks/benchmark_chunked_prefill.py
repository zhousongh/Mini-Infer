"""
Phase 9 Chunked Prefill benchmark。

测试对象：mini-infer LLMEngine（via add_request/step 接口）
测试场景：7 个短 decode 请求正在运行时，1 个长 prompt 请求到达。
          测量短请求在长请求 prefill 期间的 ITL 最大峰值（latency spike），
          以及长请求的 TTFT。

关键指标：
  - max_itl_spike_ms：短请求在长请求到达后、长请求完成 prefill 前的最大 token 间隔
    （有无 chunking 时应有显著差异，这是 chunked prefill 的核心收益指标）
  - ttft_long_ms：长请求的 TTFT（chunking 会让 TTFT 增加，这是合理的权衡）
  - 总吞吐

数据口径说明：
  - max_itl_spike 通过连续记录所有 step 的 token 时间戳计算，包含跨阶段的间隔
  - TTFT = step() 返回长请求第一个 token 时的时刻 - add_request 时刻

用法：
  export MODEL=/path/to/Qwen2.5-7B-Instruct && export HF_HUB_OFFLINE=1
  conda run -n ai-infra python benchmarks/benchmark_chunked_prefill.py --model $MODEL
  conda run -n ai-infra python benchmarks/benchmark_chunked_prefill.py --model $MODEL --chunk-size 128
"""

import argparse
import time

from mini_infer import EngineConfig, LLMEngine

NUM_DECODE_REQUESTS = 7
LONG_PROMPT_TOKENS = 1024
SHORT_PROMPT_TOKENS = 32
MAX_NEW_TOKENS_LONG = 32
MAX_NEW_TOKENS_SHORT = 128


def make_prompt_of_len(target_tokens: int) -> str:
    """生成约 target_tokens 字符的 prompt（ASCII 字符，约 1 token/char）。"""
    return "a " * target_tokens


def run_scenario(engine: LLMEngine, long_prompt: str, short_prompts: list[str]) -> dict:
    """
    完整场景（含连续 ITL 追踪）：
      1. 提交短请求，运行直到全部完成 prefill（进入 running 状态）
         — 期间记录每个短请求的 token 时间戳（建立基线 ITL）
      2. 提交长请求（记录时刻 t_long_start）
      3. 继续 step()，记录：
         a. 每个短请求的 token 时间戳（追踪 spike）
         b. 长请求的第一个 token 时刻（TTFT）
         c. 长请求首次出现后的 step 计数（标记 prefill 结束时刻）

    max_itl_spike：长请求到达后、长请求完成 prefill 前，
                   所有短请求经历的最大相邻 token 间隔（即阻塞峰值）
    """
    rid_shorts = [
        engine.add_request(p, max_new_tokens=MAX_NEW_TOKENS_SHORT)
        for p in short_prompts
    ]
    short_set = set(rid_shorts)
    tracked_rids = set(rid_shorts)

    # 连续记录 token 时间戳（全程）
    short_token_times: dict[str, list[float]] = {r: [] for r in rid_shorts}
    ttft_long: float | None = None
    total_tokens = 0
    t0 = time.perf_counter()
    prev_lens: dict[str, int] = {rid: 0 for rid in rid_shorts}

    # ── 阶段 1：warmup 直到所有短请求完成 prefill ─────────────────────────
    while not all(engine._step_states[r].prefilled for r in rid_shorts):
        engine.step()
        t_now = time.perf_counter()
        for rid in rid_shorts:
            curr_len = len(engine._step_states[rid].generated_token_ids)
            inc = curr_len - prev_lens[rid]
            if inc > 0:
                total_tokens += inc
                short_token_times[rid].extend([t_now] * inc)
                prev_lens[rid] = curr_len

    # ── 阶段 2：提交长请求 ────────────────────────────────────────────────
    t_long_start = time.perf_counter()
    rid_long = engine.add_request(long_prompt, max_new_tokens=MAX_NEW_TOKENS_LONG)
    tracked_rids.add(rid_long)
    prev_lens[rid_long] = 0

    # ── 阶段 3：运行直到所有请求完成 ──────────────────────────────────────
    while engine.has_unfinished_requests():
        engine.step()
        t_now = time.perf_counter()

        for rid in tracked_rids:
            curr_len = len(engine._step_states[rid].generated_token_ids)
            inc = curr_len - prev_lens[rid]
            if inc <= 0:
                continue
            total_tokens += inc
            prev_lens[rid] = curr_len
            if rid in short_set:
                short_token_times[rid].extend([t_now] * inc)
            elif rid == rid_long and ttft_long is None:
                ttft_long = t_now - t_long_start
                pass  # 此 step 完成了长请求的 prefill

    t_end = time.perf_counter()
    total_wall = t_end - t0

    # ── 计算 max_itl_spike：长请求到达后、完成 prefill 前的最大 ITL ────────
    # 用 t_long_start 作为"到达时刻"插入各短请求时间线，
    # 找 t_long_start 之后直到长请求 TTFT 之前（含）的最大间隔
    max_spike = 0.0
    for rid, times in short_token_times.items():
        if not times:
            continue
        # 构造完整时间序列（含 t_long_start 作为"感兴趣区间起点"）
        # 找 >= t_long_start 的首个时间点，以及之前的最近时间点
        prev_t = None
        for i, t in enumerate(times):
            if prev_t is not None:
                gap = t - prev_t
                # 只统计长请求到达后的间隔
                if prev_t >= t_long_start:
                    max_spike = max(max_spike, gap)
            if t > t_long_start and prev_t is not None and prev_t < t_long_start:
                # prev_t 在 long 到达之前，t 在之后 → 包含 long_start 的跨越间隔
                gap = t - prev_t  # 注意：也可以用 t - t_long_start，取决于业务定义
                max_spike = max(max_spike, gap)
            prev_t = t

    return {
        "ttft_long_ms": (ttft_long or 0) * 1000,
        "max_itl_spike_ms": max_spike * 1000,
        "total_tokens": total_tokens,
        "total_wall_s": total_wall,
        "throughput_toks": total_tokens / total_wall if total_wall > 0 else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 9 Chunked Prefill benchmark")
    parser.add_argument("--model", required=True, help="Qwen2.5-7B-Instruct 本地路径")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="chunk_prefill_size（默认 256）")
    parser.add_argument("--num-gpu-blocks", type=int, default=200)
    parser.add_argument("--max-batch-size", type=int, default=8)
    args = parser.parse_args()

    long_prompt = make_prompt_of_len(LONG_PROMPT_TOKENS)
    short_prompts = [make_prompt_of_len(SHORT_PROMPT_TOKENS) for _ in range(NUM_DECODE_REQUESTS)]

    print("=" * 60)
    print("Phase 9 Chunked Prefill Benchmark")
    print(f"Model: {args.model}")
    print(f"场景：7 短请求（~{SHORT_PROMPT_TOKENS}tok）decode 中，1 长请求（~{LONG_PROMPT_TOKENS}tok）到达")
    print(f"对照：chunk_prefill_size=0 vs {args.chunk_size}")
    print("=" * 60)

    base_config = dict(
        model_name=args.model,
        device=args.device,
        block_size=256,
        num_gpu_blocks=args.num_gpu_blocks,
        max_batch_size=args.max_batch_size,
    )

    # ── 场景 A：无 chunked prefill ──────────────────────────────────────
    print("\n[场景 A] chunk_prefill_size=0（无分块，Phase 8 行为）")
    engine_a = LLMEngine(EngineConfig(**base_config, chunk_prefill_size=0))
    result_a = run_scenario(engine_a, long_prompt, short_prompts)
    del engine_a

    print(f"  长请求 TTFT:                  {result_a['ttft_long_ms']:.1f} ms")
    print(f"  短请求最大 ITL spike（阻塞峰值）: {result_a['max_itl_spike_ms']:.1f} ms")
    print(f"  总吞吐:                       {result_a['throughput_toks']:.1f} tok/s")

    # ── 场景 B：chunked prefill ──────────────────────────────────────────
    print(f"\n[场景 B] chunk_prefill_size={args.chunk_size}")
    engine_b = LLMEngine(EngineConfig(**base_config, chunk_prefill_size=args.chunk_size))
    result_b = run_scenario(engine_b, long_prompt, short_prompts)
    del engine_b

    print(f"  长请求 TTFT:                  {result_b['ttft_long_ms']:.1f} ms")
    print(f"  短请求最大 ITL spike（阻塞峰值）: {result_b['max_itl_spike_ms']:.1f} ms")
    print(f"  总吞吐:                       {result_b['throughput_toks']:.1f} tok/s")

    # ── 对比 ──────────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("对比结果（B vs A）:")

    if result_a["max_itl_spike_ms"] > 0:
        spike_reduction = (
            (result_a["max_itl_spike_ms"] - result_b["max_itl_spike_ms"])
            / result_a["max_itl_spike_ms"] * 100
        )
        print(f"  短请求 ITL spike 降低:  {spike_reduction:+.1f}%  "
              f"（{result_a['max_itl_spike_ms']:.1f} → {result_b['max_itl_spike_ms']:.1f} ms）")

    ttft_change = result_b["ttft_long_ms"] - result_a["ttft_long_ms"]
    print(f"  长请求 TTFT 变化:       {ttft_change:+.1f} ms  "
          f"（{result_a['ttft_long_ms']:.1f} → {result_b['ttft_long_ms']:.1f} ms）")
    print("  （长请求 TTFT 增加是预期的：chunked prefill 以长请求延迟换短请求 ITL 平稳）")

    if result_a["throughput_toks"] > 0:
        tput_ratio = result_b["throughput_toks"] / result_a["throughput_toks"] * 100
        print(f"  吞吐保留率:             {tput_ratio:.1f}%")


if __name__ == "__main__":
    main()
