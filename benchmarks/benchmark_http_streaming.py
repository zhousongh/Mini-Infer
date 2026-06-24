"""Real HTTP streaming benchmark for mini-infer.

Unlike benchmarks/benchmark_server.py, this script talks to a real HTTP server
over a socket and consumes Server-Sent Events incrementally. That makes TTFT
measurement meaningful.

Typical usage with an already running server:
  python benchmarks/benchmark_http_streaming.py --url http://127.0.0.1:8000

Optionally launch a temporary server:
  python benchmarks/benchmark_http_streaming.py \
    --model /path/to/Qwen2.5-7B-Instruct --device cuda:0 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any

import httpx


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


@dataclass(slots=True)
class StreamingRequestResult:
    request_id: int
    concurrency: int
    status_code: int
    first_sse_ms: float | None
    ttft_ms: float | None
    total_ms: float
    content_chunks: int
    output_chars: int
    output_tokens: int | None
    mean_itl_ms: float
    p95_itl_ms: float
    finish_reason: str | None
    error: str = ""


@dataclass(slots=True)
class ConcurrencySummary:
    concurrency: int
    requests: int
    ok: int
    failed: int
    mean_ttft_ms: float
    p50_ttft_ms: float
    p95_ttft_ms: float
    mean_latency_ms: float
    p95_latency_ms: float
    mean_itl_ms: float
    p95_itl_ms: float
    total_output_tokens: int | None
    total_output_chars: int
    tokens_per_s: float | None
    chars_per_s: float


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int((len(ordered) - 1) * pct))
    return float(ordered[idx])


def parse_sse_data(data: str) -> tuple[str, str | None, str | None]:
    """Return (kind, content, finish_reason) for one OpenAI-style SSE data item."""
    if data == "[DONE]":
        return "done", None, None
    payload = json.loads(data)
    choice = payload["choices"][0]
    delta = choice.get("delta") or {}
    content = delta.get("content")
    finish_reason = choice.get("finish_reason")
    if content:
        return "content", str(content), finish_reason
    if finish_reason:
        return "finish", None, str(finish_reason)
    return "other", None, None


async def measure_streaming_request(
    client: httpx.AsyncClient,
    url: str,
    request_id: int,
    concurrency: int,
    prompt: str,
    model: str,
    max_tokens: int,
    tokenizer: Any | None,
) -> StreamingRequestResult:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }

    t0 = time.perf_counter()
    first_sse_ms: float | None = None
    ttft_ms: float | None = None
    chunk_times: list[float] = []
    text_parts: list[str] = []
    finish_reason: str | None = None
    status_code = 0

    try:
        async with client.stream("POST", f"{url}/v1/chat/completions", json=payload) as resp:
            status_code = resp.status_code
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                now = time.perf_counter()
                if first_sse_ms is None:
                    first_sse_ms = (now - t0) * 1000
                kind, content, reason = parse_sse_data(line[6:])
                if reason is not None:
                    finish_reason = reason
                if kind == "done":
                    break
                if content:
                    text_parts.append(content)
                    chunk_times.append(now)
                    if ttft_ms is None:
                        ttft_ms = (now - t0) * 1000
        total_ms = (time.perf_counter() - t0) * 1000
    except Exception as exc:  # noqa: BLE001 - benchmark should report per-request errors.
        total_ms = (time.perf_counter() - t0) * 1000
        return StreamingRequestResult(
            request_id=request_id,
            concurrency=concurrency,
            status_code=status_code,
            first_sse_ms=first_sse_ms,
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            content_chunks=len(chunk_times),
            output_chars=sum(len(part) for part in text_parts),
            output_tokens=None,
            mean_itl_ms=0.0,
            p95_itl_ms=0.0,
            finish_reason=finish_reason,
            error=str(exc),
        )

    itls = [
        (chunk_times[i] - chunk_times[i - 1]) * 1000
        for i in range(1, len(chunk_times))
    ]
    output_text = "".join(text_parts)
    output_tokens = None
    if tokenizer is not None:
        output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))

    return StreamingRequestResult(
        request_id=request_id,
        concurrency=concurrency,
        status_code=status_code,
        first_sse_ms=first_sse_ms,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        content_chunks=len(chunk_times),
        output_chars=len(output_text),
        output_tokens=output_tokens,
        mean_itl_ms=statistics.mean(itls) if itls else 0.0,
        p95_itl_ms=percentile(itls, 0.95),
        finish_reason=finish_reason,
    )


async def run_concurrency(
    *,
    url: str,
    model: str,
    concurrency: int,
    max_tokens: int,
    timeout_s: float,
    tokenizer: Any | None,
) -> tuple[ConcurrencySummary, list[StreamingRequestResult]]:
    async with httpx.AsyncClient(timeout=timeout_s, trust_env=False) as client:
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(concurrency)]
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *[
                measure_streaming_request(
                    client=client,
                    url=url,
                    request_id=i,
                    concurrency=concurrency,
                    prompt=prompt,
                    model=model,
                    max_tokens=max_tokens,
                    tokenizer=tokenizer,
                )
                for i, prompt in enumerate(prompts)
            ]
        )
        wall_s = time.perf_counter() - t0

    ok_results = [r for r in results if not r.error and r.ttft_ms is not None]
    ttfts = [float(r.ttft_ms) for r in ok_results if r.ttft_ms is not None]
    latencies = [r.total_ms for r in ok_results]
    mean_itls = [r.mean_itl_ms for r in ok_results if r.mean_itl_ms > 0]
    p95_itls = [r.p95_itl_ms for r in ok_results if r.p95_itl_ms > 0]
    output_chars = sum(r.output_chars for r in ok_results)
    token_values = [r.output_tokens for r in ok_results if r.output_tokens is not None]
    total_tokens = sum(token_values) if len(token_values) == len(ok_results) and ok_results else None

    summary = ConcurrencySummary(
        concurrency=concurrency,
        requests=len(results),
        ok=len(ok_results),
        failed=len(results) - len(ok_results),
        mean_ttft_ms=statistics.mean(ttfts) if ttfts else 0.0,
        p50_ttft_ms=percentile(ttfts, 0.50),
        p95_ttft_ms=percentile(ttfts, 0.95),
        mean_latency_ms=statistics.mean(latencies) if latencies else 0.0,
        p95_latency_ms=percentile(latencies, 0.95),
        mean_itl_ms=statistics.mean(mean_itls) if mean_itls else 0.0,
        p95_itl_ms=percentile(p95_itls, 0.95),
        total_output_tokens=total_tokens,
        total_output_chars=output_chars,
        tokens_per_s=(total_tokens / wall_s) if total_tokens is not None and wall_s > 0 else None,
        chars_per_s=(output_chars / wall_s) if wall_s > 0 else 0.0,
    )
    return summary, results


def load_tokenizer(path: str) -> Any | None:
    if not path:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return tokenizer


def wait_for_health(url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    last_error = ""
    with httpx.Client(timeout=2.0, trust_env=False) as client:
        while time.time() < deadline:
            try:
                resp = client.get(f"{url}/healthz")
                if resp.status_code == 200 and resp.json().get("status") == "ok":
                    return
                last_error = resp.text
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
            time.sleep(1.0)
    raise RuntimeError(f"server did not become healthy within {timeout_s:.1f}s: {last_error}")


def start_server(args: argparse.Namespace) -> subprocess.Popen[bytes] | None:
    if not args.model and not args.dry_run:
        return None
    binary = shutil.which("mini-infer-serve")
    if binary is None:
        raise RuntimeError("mini-infer-serve not found in PATH")

    cmd = [
        binary,
        "--host", args.host,
        "--port", str(args.port),
        "--device", args.device,
        "--dtype", args.dtype,
        "--max-batch-size", str(args.max_batch_size),
        "--num-gpu-blocks", str(args.num_gpu_blocks),
        "--block-size", str(args.block_size),
        "--chunk-prefill-size", str(args.chunk_prefill_size),
        "--scheduler-policy", args.scheduler_policy,
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    else:
        cmd.extend(["--model", args.model])

    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")

    proc = subprocess.Popen(  # noqa: S603 - benchmark launches local console script.
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc


def stop_server(proc: subprocess.Popen[bytes] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=5)


def print_summary_table(summaries: list[ConcurrencySummary]) -> None:
    print("\nReal HTTP streaming benchmark")
    print("  TTFT is first content SSE chunk arrival, not full-response latency.")
    print()
    print(
        "  "
        f"{'conc':>4}  {'ok':>5}  {'fail':>5}  "
        f"{'ttft_mean':>10}  {'ttft_p95':>9}  "
        f"{'lat_mean':>9}  {'lat_p95':>8}  "
        f"{'itl_mean':>9}  {'itl_p95':>8}  "
        f"{'tok/s':>8}  {'chars/s':>8}"
    )
    print("  " + "-" * 101)
    for s in summaries:
        tok_s = "-" if s.tokens_per_s is None else f"{s.tokens_per_s:.1f}"
        print(
            "  "
            f"{s.concurrency:>4}  {s.ok:>5}  {s.failed:>5}  "
            f"{s.mean_ttft_ms:>10.1f}  {s.p95_ttft_ms:>9.1f}  "
            f"{s.mean_latency_ms:>9.1f}  {s.p95_latency_ms:>8.1f}  "
            f"{s.mean_itl_ms:>9.1f}  {s.p95_itl_ms:>8.1f}  "
            f"{tok_s:>8}  {s.chars_per_s:>8.1f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real HTTP streaming TTFT benchmark")
    parser.add_argument("--url", default="", help="existing server URL, e.g. http://127.0.0.1:8000")
    parser.add_argument("--model-id", default="mini-infer", help="OpenAI model id in request JSON")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--tokenizer", default="", help="optional tokenizer path for token/s metrics")

    # If --model or --dry-run is set, this script launches a temporary server.
    parser.add_argument("--model", default="", help="optional model path; launches mini-infer-serve")
    parser.add_argument("--dry-run", action="store_true", help="launch dry-run server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--num-gpu-blocks", type=int, default=200)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--chunk-prefill-size", type=int, default=0)
    parser.add_argument(
        "--scheduler-policy",
        default="pressure_aware",
        choices=["baseline", "reserve_only", "pressure_aware", "adaptive", "slo_aware"],
    )
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    return parser.parse_args()


async def async_main(args: argparse.Namespace) -> dict[str, Any]:
    url = args.url.rstrip("/")
    if not url:
        url = f"http://{args.host}:{args.port}"

    tokenizer_path = args.tokenizer or ""
    tokenizer = load_tokenizer(tokenizer_path)

    async with httpx.AsyncClient(timeout=args.timeout, trust_env=False) as client:
        for _ in range(args.warmup):
            await measure_streaming_request(
                client=client,
                url=url,
                request_id=-1,
                concurrency=1,
                prompt=PROMPTS[0],
                model=args.model_id,
                max_tokens=min(args.max_tokens, 8),
                tokenizer=tokenizer,
            )

    summaries: list[ConcurrencySummary] = []
    all_results: list[StreamingRequestResult] = []
    for concurrency in args.concurrency:
        summary, results = await run_concurrency(
            url=url,
            model=args.model_id,
            concurrency=concurrency,
            max_tokens=args.max_tokens,
            timeout_s=args.timeout,
            tokenizer=tokenizer,
        )
        summaries.append(summary)
        all_results.extend(results)

    return {
        "url": url,
        "model_id": args.model_id,
        "max_tokens": args.max_tokens,
        "summaries": [asdict(summary) for summary in summaries],
        "requests": [asdict(result) for result in all_results],
    }


def main() -> None:
    args = parse_args()
    proc = start_server(args)
    try:
        if proc is not None:
            url = args.url.rstrip("/") if args.url else f"http://{args.host}:{args.port}"
            wait_for_health(url, args.startup_timeout)
        result = asyncio.run(async_main(args))
    finally:
        stop_server(proc)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print_summary_table([
            ConcurrencySummary(**summary)
            for summary in result["summaries"]
        ])


if __name__ == "__main__":
    main()
