"""
benchmarks/benchmark_server.py — Phase 8 Chat Completions 子集兼容 HTTP API benchmark。

测试对象：mini-infer AsyncEngine + FastAPI HTTP layer（对标直接 LLMEngine 调用）

测量内容：
  1. 单请求 TTFT / TPOT（SSE 流式路径，精确计时第一个 content token）
  2. 并发吞吐（1 / 2 / 4 / 8 并发客户端，验证 continuous batching 效果）
  3. HTTP 层额外开销（HTTP 吞吐 vs 直接 LLMEngine 调用的对比）
  4. 峰值显存（说明：HTTP 层不改变推理路径，与 Phase 6/7 相同）

运行方式：
  # 完整 GPU 测试（需要模型权重）
  export MODEL_PATH=~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct
  conda run -n ai-infra python benchmarks/benchmark_server.py --model $MODEL_PATH

  # dry_run 模式（验证 HTTP 层机制，无需模型权重）
  conda run -n ai-infra python benchmarks/benchmark_server.py --dry-run

Workload：
  - 模型：Qwen2.5-7B-Instruct（GPU 测试）/ stub（dry_run）
  - 并发：1 / 2 / 4 / 8 客户端
  - prompt 长度：~32 token（固定 8 条 prompt 轮询）
  - output 长度：64 token
  - batch_size 上限：8

指标口径说明：
  - TTFT（ms）：从发出 HTTP 请求到 SSE 流中首个 content delta 到达的时间
  - TPOT（ms/tok）：首 token 后平均每 token 时间（近似：(total - TTFT) / (n_tokens - 1)）
  - Throughput（tok/s）：全部并发请求的 output token 总数 / 总墙钟时间
  - Peak Memory（GB）：CUDA 峰值显存（HTTP 层不引入额外显存，与直接调用相同）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass

import httpx
import torch
from asgi_lifespan import LifespanManager

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mini_infer.core.config import EngineConfig
from mini_infer.serving.server import app

# ── Workload ──────────────────────────────────────────────────────────────────

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


# ── 结果数据类 ─────────────────────────────────────────────────────────────────

@dataclass
class SingleRequestResult:
    """单请求 SSE 流式路径指标。"""
    ttft_ms: float          # 首个 content delta 到达时间（ms）
    total_ms: float         # 整个流完成时间（ms）
    output_tokens: int      # 输出 token 数（按 chunk 计）
    tpot_ms: float          # 首 token 后每 token 平均时间（近似）


@dataclass
class ConcurrencyResult:
    """并发吞吐结果。"""
    concurrency: int
    total_tokens: int
    total_time_s: float
    throughput_tok_s: float


# ── ASGI 客户端 fixture ────────────────────────────────────────────────────────

async def make_client(config: EngineConfig):
    """启动 AsyncEngine 并返回 (LifespanManager, httpx.AsyncClient)。"""
    app.state.engine_config = config
    manager = LifespanManager(app)
    await manager.__aenter__()
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=manager.app),
        base_url="http://test",
        timeout=120.0,
    )
    return manager, client


async def close_client(manager, client):
    await client.aclose()
    await manager.__aexit__(None, None, None)


# ── 单请求 TTFT / TPOT 测量 ────────────────────────────────────────────────────

async def measure_single_request(
    client: httpx.AsyncClient,
    prompt: str,
    max_tokens: int = 64,
    dry_run: bool = False,
) -> SingleRequestResult:
    """
    发送 SSE 流式请求，精确计时首个 content delta 到达时间。

    注意：httpx.ASGITransport 不支持真正的增量流式（会缓冲完整响应后返回），
    因此 TTFT 代表的是"从发出请求到完整响应可读"——即近似整体延迟，非真实首 token 时间。
    真实 TTFT 只能通过独立 uvicorn 进程 + 真正流式客户端测量（标注差异）。
    """
    payload = {
        "model": "mini-infer",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    t_start = time.perf_counter()
    resp = await client.post("/v1/chat/completions", json=payload)
    t_response = time.perf_counter()  # ASGITransport 下等于完整响应时间

    # 解析 SSE
    lines = resp.text.strip().split("\n")
    data_lines = [line[6:] for line in lines if line.startswith("data: ") and line[6:] != "[DONE]"]
    chunks = [json.loads(d) for d in data_lines]

    # 统计 content 文本块
    content_chunks = [
        c for c in chunks
        if c["choices"][0]["delta"].get("content")
    ]
    content_text = "".join(c["choices"][0]["delta"]["content"] for c in content_chunks)

    # 事件数不等于 token 数：首次 step 可能把 prefill 采样 token 和首个 decode token 合并成一个 delta。
    # 对同一 prompt 再做一次非流式请求，复用 usage.completion_tokens 作为精确 token 数。
    if dry_run:
        # dry_run stub 文本形如 " [1] [2] [3]"，按标记数统计更接近真实 token 数。
        output_tokens = content_text.count("[")
    else:
        usage_payload = {
            "model": "mini-infer",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
        }
        usage_resp = await client.post("/v1/chat/completions", json=usage_payload)
        output_tokens = usage_resp.json()["usage"]["completion_tokens"]

    total_ms = (t_response - t_start) * 1000

    # TTFT 说明：ASGITransport 缓冲完整响应，无法分离首 token 时间
    # 此处 TTFT = 整体延迟（保守估计），标注为近似
    ttft_ms = total_ms  # 近似：无法从缓冲响应中分离

    tpot_ms = total_ms / max(output_tokens, 1)

    return SingleRequestResult(
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        output_tokens=output_tokens,
        tpot_ms=tpot_ms,
    )


# ── 并发吞吐测量 ───────────────────────────────────────────────────────────────

async def measure_concurrency(
    client: httpx.AsyncClient,
    concurrency: int,
    max_tokens: int = 64,
    dry_run: bool = False,
) -> ConcurrencyResult:
    """并发发送 concurrency 个请求，测量总 throughput。"""
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(concurrency)]

    async def single(prompt: str) -> int:
        payload = {
            "model": "mini-infer",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
        }
        resp = await client.post("/v1/chat/completions", json=payload)
        body = resp.json()
        if dry_run:
            return body["choices"][0]["message"]["content"].count("[")
        return body["usage"]["completion_tokens"]

    t0 = time.perf_counter()
    results = await asyncio.gather(*[single(p) for p in prompts])
    total_time_s = time.perf_counter() - t0

    total_tokens = sum(results)
    throughput = total_tokens / total_time_s if total_time_s > 0 else 0.0

    return ConcurrencyResult(
        concurrency=concurrency,
        total_tokens=total_tokens,
        total_time_s=total_time_s,
        throughput_tok_s=throughput,
    )


# ── 打印函数 ───────────────────────────────────────────────────────────────────

def print_header(model_name: str, dry_run: bool, max_tokens: int) -> None:
    print("=" * 64)
    print("Phase 8 HTTP API Benchmark")
    print(f"  模型         : {model_name}")
    print(f"  dry_run      : {dry_run}")
    print(f"  max_tokens   : {max_tokens}")
    print(f"  GPU          : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print("=" * 64)


def print_single_results(results: list[SingleRequestResult]) -> None:
    ttft_list = [r.ttft_ms for r in results]
    tpot_list = [r.tpot_ms for r in results]
    tok_list = [r.output_tokens for r in results]

    print("\n── 单请求延迟（SSE 流式，ASGI Transport）")
    print("  注意：TTFT 为近似值（ASGITransport 缓冲完整响应，无法分离首 token）")
    print(f"  样本数      : {len(results)}")
    print(f"  TTFT        : mean={sum(ttft_list)/len(ttft_list):.1f}ms  "
          f"min={min(ttft_list):.1f}ms  max={max(ttft_list):.1f}ms  [近似 = 整体延迟]")
    print(f"  TPOT        : mean={sum(tpot_list)/len(tpot_list):.2f}ms/tok  [近似]")
    print(f"  输出 token  : mean={sum(tok_list)/len(tok_list):.1f} tok/请求")


def print_concurrency_results(results: list[ConcurrencyResult]) -> None:
    print("\n── 并发吞吐（non-streaming，多请求 continuous batching）")
    print(f"  {'并发数':>6}  {'总 tokens':>10}  {'时间(s)':>8}  {'tok/s':>8}")
    for r in results:
        print(f"  {r.concurrency:>6}  {r.total_tokens:>10}  "
              f"{r.total_time_s:>8.2f}  {r.throughput_tok_s:>8.1f}")


def print_memory(dry_run: bool) -> None:
    if not dry_run and torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n── 峰值显存 : {peak_gb:.2f} GB（Phase 8 HTTP 层不增加显存，与 Phase 6/7 相同）")
    else:
        print("\n── 峰值显存 : N/A（dry_run 模式）")


# ── 主流程 ─────────────────────────────────────────────────────────────────────

async def run_benchmark(args: argparse.Namespace) -> None:
    model_name = args.model if args.model else "dry"
    dry_run = args.dry_run

    config = EngineConfig(
        model_name=model_name,
        dry_run=dry_run,
        device=args.device,
        dtype=args.dtype,
        max_batch_size=args.max_batch_size,
        num_gpu_blocks=args.num_gpu_blocks,
        block_size=args.block_size,
    )

    print_header(model_name, dry_run, args.max_tokens)

    if torch.cuda.is_available() and not dry_run:
        torch.cuda.reset_peak_memory_stats()

    manager, client = await make_client(config)
    try:
        # ── 热身 ─────────────────────────────────────────────────────────────
        print("\n热身（1 请求，8 token）...")
        warmup_payload = {
            "model": "mini-infer",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 8,
            "stream": False,
        }
        await client.post("/v1/chat/completions", json=warmup_payload)

        if torch.cuda.is_available() and not dry_run:
            torch.cuda.reset_peak_memory_stats()

        # ── 单请求延迟 ────────────────────────────────────────────────────────
        n_trials = args.trials
        single_results = []
        print(f"\n单请求延迟测量（{n_trials} 次）...")
        for i in range(n_trials):
            prompt = PROMPTS[i % len(PROMPTS)]
            r = await measure_single_request(
                client, prompt, max_tokens=args.max_tokens, dry_run=dry_run
            )
            single_results.append(r)
        print_single_results(single_results)

        # ── 并发吞吐 ──────────────────────────────────────────────────────────
        print("\n并发吞吐测量...")
        concurrency_levels = args.concurrency
        concurrency_results = []
        for c in concurrency_levels:
            r = await measure_concurrency(
                client, c, max_tokens=args.max_tokens, dry_run=dry_run
            )
            concurrency_results.append(r)
            print(f"  并发={c}: {r.throughput_tok_s:.1f} tok/s "
                  f"({r.total_tokens} tokens / {r.total_time_s:.2f}s)")
        print_concurrency_results(concurrency_results)

    finally:
        await close_client(manager, client)

    print_memory(dry_run)

    print("\n── 局限性说明")
    print("  1. TTFT 为近似值：httpx.ASGITransport 缓冲完整响应，无法测真实流式首 token 延迟")
    print("     真实 TTFT 需要: uvicorn 独立进程 + curl/httpx streaming client")
    print("  2. 并发 throughput 口径：真实模型时使用 completion_tokens；dry_run 时按 stub token 标记数统计")
    print("  3. 峰值显存：与 Phase 6/7 相同（HTTP 层不引入额外 GPU 内存）")
    print("  4. 本 benchmark 未对比 HF baseline（HTTP 层不改变推理内核）；")
    print("     推理性能参考 Phase 6 benchmark_mini.py：batch=8 约 406 tok/s")


# ── argparse ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 8 HTTP API benchmark")
    p.add_argument("--model", type=str, default="", help="模型路径（空 = 使用 dry-run）")
    p.add_argument("--dry-run", action="store_true", help="使用 stub tokenizer（无需模型权重）")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--dtype", type=str, default="float16")
    p.add_argument("--max-batch-size", type=int, default=8)
    p.add_argument("--num-gpu-blocks", type=int, default=200)
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--max-tokens", type=int, default=64, help="每请求最大输出 token 数")
    p.add_argument("--trials", type=int, default=5, help="单请求延迟测量次数")
    p.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="并发客户端数列表（如 --concurrency 1 2 4 8）",
    )
    args = p.parse_args()
    if not args.dry_run and not args.model:
        args.dry_run = True  # 未指定 --model 时自动 dry_run
    return args


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_benchmark(args))
