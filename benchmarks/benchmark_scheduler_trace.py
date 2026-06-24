"""
benchmark_scheduler_trace.py — Synthetic scheduler trace benchmark.

This benchmark is dry-run only. It generates a reproducible request trace with
controlled prompt length, output length, priority, and arrival step. The goal is
to evaluate scheduler behavior under realistic-ish KV pressure before and after
policy changes.

Run:
  python benchmarks/benchmark_scheduler_trace.py
  python benchmarks/benchmark_scheduler_trace.py --json
  python benchmarks/benchmark_scheduler_trace.py --dump-trace /tmp/trace.jsonl

The generated prompts are synthetic strings. In dry-run mode, one character maps
to one token, so prompt length is exact and easy to reason about.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from mini_infer import EngineConfig, LLMEngine


@dataclass(slots=True)
class TraceRequest:
    request_id: str
    arrival_step: int
    prompt_len: int
    max_new_tokens: int
    priority: int
    bucket: str

    @property
    def prompt(self) -> str:
        # dry_run tokenizer maps each character to one token.
        return self.bucket[0].upper() * self.prompt_len


@dataclass(slots=True)
class TraceResult:
    scheduler_policy: str
    seed: int
    num_requests: int
    status: str
    steps: int
    elapsed_ms: float
    completed: int
    failed: int
    output_tokens: int
    throughput_tok_per_step: float
    peak_kv_utilization: float
    avg_kv_utilization: float
    avg_waiting: float
    max_waiting: int
    avg_running: float
    max_running: int
    max_swapped: int
    mean_ttft_steps: float
    p95_ttft_steps: float
    ttft_slo_steps: int
    ttft_slo_miss_rate: float
    latency_slo_steps: int
    latency_slo_miss_rate: float
    mean_latency_steps: float
    p95_latency_steps: float
    decode_oom_count: int
    admission_error_count: int
    counters: dict[str, int]
    bucket_counts: dict[str, int]
    error: str = ""


def generate_trace(
    *,
    seed: int,
    num_requests: int,
    max_arrival_gap: int,
) -> list[TraceRequest]:
    rng = random.Random(seed)
    arrival_step = 0
    trace: list[TraceRequest] = []

    for idx in range(num_requests):
        arrival_step += rng.randint(0, max_arrival_gap)
        p = rng.random()
        if p < 0.70:
            bucket = "short"
            prompt_len = rng.randint(4, 16)
            max_new_tokens = rng.randint(4, 16)
        elif p < 0.95:
            bucket = "medium"
            prompt_len = rng.randint(32, 96)
            max_new_tokens = rng.randint(16, 32)
        else:
            bucket = "long"
            prompt_len = rng.randint(128, 256)
            max_new_tokens = rng.randint(16, 64)

        q = rng.random()
        if q < 0.05:
            priority = 0      # urgent
        elif q < 0.20:
            priority = 1      # high
        else:
            priority = 3      # normal

        trace.append(
            TraceRequest(
                request_id=f"req-{idx:04d}",
                arrival_step=arrival_step,
                prompt_len=prompt_len,
                max_new_tokens=max_new_tokens,
                priority=priority,
                bucket=bucket,
            )
        )

    return trace


def make_engine(args: argparse.Namespace) -> LLMEngine:
    return LLMEngine(
        EngineConfig(
            model_name="stub",
            dry_run=True,
            block_size=args.block_size,
            num_gpu_blocks=args.num_gpu_blocks,
            max_batch_size=args.max_batch_size,
            chunk_prefill_size=args.chunk_prefill_size,
            scheduler_policy=args.scheduler_policy,
            adaptive_reserve_threshold=args.adaptive_reserve_threshold,
            adaptive_preempt_threshold=args.adaptive_preempt_threshold,
            adaptive_waiting_threshold=args.adaptive_waiting_threshold,
            scheduler_ttft_slo_steps=args.ttft_slo_steps,
            scheduler_latency_slo_steps=args.latency_slo_steps,
        )
    )


def percentile(values: list[int], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int((len(ordered) - 1) * pct))
    return float(ordered[idx])


def dump_trace(trace: list[TraceRequest], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for req in trace:
            f.write(json.dumps(asdict(req), ensure_ascii=False) + "\n")


def run_trace(args: argparse.Namespace, trace: list[TraceRequest]) -> TraceResult:
    engine = make_engine(args)
    arrivals = sorted(trace, key=lambda req: (req.arrival_step, req.request_id))
    next_arrival = 0
    step = 0
    output_tokens = 0
    decode_oom_count = 0
    admission_error_count = 0
    error = ""
    status = "ok"
    start_by_request: dict[str, int] = {}
    first_token_by_request: dict[str, int] = {}
    finish_by_request: dict[str, int] = {}
    active_request_ids: set[str] = set()
    kv_utils: list[float] = []
    waiting_sizes: list[int] = []
    running_sizes: list[int] = []
    swapped_sizes: list[int] = []

    def observe() -> None:
        kv_stats = engine.kv_cache.stats()
        sched_stats = engine.scheduler.stats()
        kv_utils.append(float(kv_stats["utilization"]))
        waiting_sizes.append(int(sched_stats["waiting"]))
        running_sizes.append(int(sched_stats["running"]))
        swapped_sizes.append(int(sched_stats["swapped"]))

    t0 = time.perf_counter()
    while next_arrival < len(arrivals) or engine.has_unfinished_requests():
        while (
            next_arrival < len(arrivals)
            and arrivals[next_arrival].arrival_step <= step
        ):
            req = arrivals[next_arrival]
            engine.add_request(
                req.prompt,
                max_new_tokens=req.max_new_tokens,
                priority=req.priority,
                request_id=req.request_id,
            )
            start_by_request[req.request_id] = step
            active_request_ids.add(req.request_id)
            next_arrival += 1

        observe()

        if engine.has_unfinished_requests():
            try:
                tokens_by_request = engine.step()
            except RuntimeError as exc:
                error = str(exc)
                if "KV cache 已满" in error:
                    decode_oom_count += 1
                    status = "decode_oom"
                else:
                    admission_error_count += 1
                    status = "admission_error"
                observe()
                break

            output_tokens += sum(len(tokens) for tokens in tokens_by_request.values())
            for request_id, tokens in tokens_by_request.items():
                if tokens and request_id not in first_token_by_request:
                    first_token_by_request[request_id] = step

            for request_id in list(active_request_ids):
                if engine.is_finished(request_id):
                    finish_by_request[request_id] = step
                    engine.cleanup_request(request_id)
                    active_request_ids.remove(request_id)

            step += 1
        elif next_arrival < len(arrivals):
            step = max(step + 1, arrivals[next_arrival].arrival_step)

    elapsed_ms = (time.perf_counter() - t0) * 1000
    counters = engine.scheduler.stats()["counters"]
    completed = len(finish_by_request)
    failed = len(trace) - completed
    latencies = [
        finish_by_request[rid] - start_by_request[rid] + 1
        for rid in finish_by_request
    ]
    ttfts = [
        first_token_by_request[rid] - start_by_request[rid] + 1
        for rid in first_token_by_request
        if rid in start_by_request
    ]
    ttft_slo_misses = sum(1 for value in ttfts if value > args.ttft_slo_steps)
    latency_slo_misses = sum(
        1 for value in latencies if value > args.latency_slo_steps
    )
    bucket_counts: dict[str, int] = {"short": 0, "medium": 0, "long": 0}
    for req in trace:
        bucket_counts[req.bucket] += 1

    return TraceResult(
        scheduler_policy=args.scheduler_policy,
        seed=args.seed,
        num_requests=len(trace),
        status=status,
        steps=step,
        elapsed_ms=elapsed_ms,
        completed=completed,
        failed=failed,
        output_tokens=output_tokens,
        throughput_tok_per_step=(output_tokens / step) if step > 0 else 0.0,
        peak_kv_utilization=max(kv_utils) if kv_utils else 0.0,
        avg_kv_utilization=statistics.mean(kv_utils) if kv_utils else 0.0,
        avg_waiting=statistics.mean(waiting_sizes) if waiting_sizes else 0.0,
        max_waiting=max(waiting_sizes) if waiting_sizes else 0,
        avg_running=statistics.mean(running_sizes) if running_sizes else 0.0,
        max_running=max(running_sizes) if running_sizes else 0,
        max_swapped=max(swapped_sizes) if swapped_sizes else 0,
        mean_ttft_steps=statistics.mean(ttfts) if ttfts else 0.0,
        p95_ttft_steps=percentile(ttfts, 0.95),
        ttft_slo_steps=args.ttft_slo_steps,
        ttft_slo_miss_rate=(ttft_slo_misses / len(ttfts)) if ttfts else 0.0,
        latency_slo_steps=args.latency_slo_steps,
        latency_slo_miss_rate=(
            latency_slo_misses / len(latencies)
        ) if latencies else 0.0,
        mean_latency_steps=statistics.mean(latencies) if latencies else 0.0,
        p95_latency_steps=percentile(latencies, 0.95),
        decode_oom_count=decode_oom_count,
        admission_error_count=admission_error_count,
        counters=counters,
        bucket_counts=bucket_counts,
        error=error,
    )


def print_result(result: TraceResult) -> None:
    c = result.counters
    print("Synthetic scheduler trace")
    print(f"  scheduler_policy     : {result.scheduler_policy}")
    print(f"  seed                 : {result.seed}")
    print(f"  requests             : {result.num_requests} {result.bucket_counts}")
    print(f"  status               : {result.status}")
    print(f"  completed / failed   : {result.completed} / {result.failed}")
    print(f"  steps                : {result.steps}")
    print(f"  output_tokens        : {result.output_tokens}")
    print(f"  throughput           : {result.throughput_tok_per_step:.2f} tok/step")
    print(f"  KV utilization       : peak={result.peak_kv_utilization:.2f}  avg={result.avg_kv_utilization:.2f}")
    print(f"  waiting queue        : max={result.max_waiting}  avg={result.avg_waiting:.2f}")
    print(f"  running queue        : max={result.max_running}  avg={result.avg_running:.2f}")
    print(f"  swapped max          : {result.max_swapped}")
    print(f"  TTFT steps           : mean={result.mean_ttft_steps:.2f}  p95={result.p95_ttft_steps:.2f}")
    print(f"  TTFT SLO miss        : {result.ttft_slo_miss_rate:.2%}  [SLO={result.ttft_slo_steps} steps]")
    print(f"  latency steps        : mean={result.mean_latency_steps:.2f}  p95={result.p95_latency_steps:.2f}")
    print(f"  latency SLO miss     : {result.latency_slo_miss_rate:.2%}  [SLO={result.latency_slo_steps} steps]")
    print(f"  decode_oom_count     : {result.decode_oom_count}")
    print(f"  admission_error_count: {result.admission_error_count}")
    print("  scheduler counters")
    print(f"    admit              : {c['admit_count']}")
    print(f"    reject             : {c['reject_count']}")
    print(f"    preempt            : {c['preempt_count']}")
    print(f"    un_admit           : {c['un_admit_count']}")
    print(f"    swap_out           : {c['swap_out_count']}")
    print(f"    swap_in            : {c['swap_in_count']}")
    print(f"    finish             : {c['finish_count']}")
    if result.error:
        print(f"  error                : {result.error}")


def print_compare(results: list[TraceResult]) -> None:
    headers = [
        "policy",
        "status",
        "completed",
        "failed",
        "decode_oom",
        "steps",
        "tok/step",
        "ttft_p95",
        "ttft_miss",
        "p95",
        "lat_miss",
        "preempt",
        "swap_out",
        "swap_in",
    ]
    rows = []
    for result in results:
        c = result.counters
        rows.append([
            result.scheduler_policy,
            result.status,
            str(result.completed),
            str(result.failed),
            str(result.decode_oom_count),
            str(result.steps),
            f"{result.throughput_tok_per_step:.2f}",
            f"{result.p95_ttft_steps:.0f}",
            f"{result.ttft_slo_miss_rate:.0%}",
            f"{result.p95_latency_steps:.0f}",
            f"{result.latency_slo_miss_rate:.0%}",
            str(c["preempt_count"]),
            str(c["swap_out_count"]),
            str(c["swap_in_count"]),
        ])

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    print("  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthetic scheduler trace benchmark")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--max-arrival-gap", type=int, default=2)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-gpu-blocks", type=int, default=64)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--chunk-prefill-size", type=int, default=0)
    parser.add_argument(
        "--scheduler-policy",
        choices=["baseline", "reserve_only", "pressure_aware", "adaptive", "slo_aware"],
        default="pressure_aware",
    )
    parser.add_argument("--adaptive-reserve-threshold", type=float, default=0.60)
    parser.add_argument("--adaptive-preempt-threshold", type=float, default=0.85)
    parser.add_argument("--adaptive-waiting-threshold", type=int, default=8)
    parser.add_argument("--ttft-slo-steps", type=int, default=16)
    parser.add_argument("--latency-slo-steps", type=int, default=64)
    parser.add_argument(
        "--compare-policies",
        action="store_true",
        help="run baseline, reserve_only, pressure_aware, adaptive, and slo_aware on the same trace",
    )
    parser.add_argument("--dump-trace", type=str, default="")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace = generate_trace(
        seed=args.seed,
        num_requests=args.num_requests,
        max_arrival_gap=args.max_arrival_gap,
    )
    if args.dump_trace:
        dump_trace(trace, args.dump_trace)

    if args.compare_policies:
        results = []
        for policy in ["baseline", "reserve_only", "pressure_aware", "adaptive", "slo_aware"]:
            args.scheduler_policy = policy
            results.append(run_trace(args, trace))
        if args.json:
            print(json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False))
        else:
            print_compare(results)
        return

    result = run_trace(args, trace)
    if args.json:
        print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    else:
        print_result(result)


if __name__ == "__main__":
    main()
