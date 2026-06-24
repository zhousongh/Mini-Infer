"""
benchmark_scheduler_counters.py — Scheduler counter pressure tests.

This benchmark is intentionally dry-run only. It validates scheduler observability
without loading model weights or allocating GPU KV tensors.

Scenarios:
  1. no_pressure: enough KV blocks; requests finish normally.
  2. un_admit_pressure: multiple fresh requests compete for too few blocks.
  3. swap_pressure: a high-priority request arrives after a low-priority request
     has already been prefilling/decoding, forcing swap_out and swap_in.
  4. reject_oversize: one request needs more KV blocks than the whole pool.

Workload scenarios:
  5. short_long_mix: one long request fills the KV pool; same-priority short
     requests arrive later and must wait.
  6. priority_late_arrival: a high-priority request arrives after a low-priority
     request is running, forcing swap_out and later swap_in.
  7. kv_oversubscription: many requests collectively exceed the KV pool, so the
     scheduler serializes admission under sustained pressure.

Run:
  python benchmarks/benchmark_scheduler_counters.py
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Callable

from mini_infer import EngineConfig, LLMEngine

SCHEDULER_POLICY = "pressure_aware"


@dataclass(slots=True)
class ScenarioResult:
    name: str
    status: str
    steps: int
    output_tokens: int
    elapsed_ms: float
    peak_kv_utilization: float
    max_waiting: int
    max_running: int
    max_swapped: int
    counters: dict[str, int]
    error: str = ""


@dataclass(slots=True)
class RunStats:
    steps: int = 0
    output_tokens: int = 0
    peak_kv_utilization: float = 0.0
    max_waiting: int = 0
    max_running: int = 0
    max_swapped: int = 0

    def observe(self, engine: LLMEngine) -> None:
        kv_stats = engine.kv_cache.stats()
        sched_stats = engine.scheduler.stats()
        self.peak_kv_utilization = max(
            self.peak_kv_utilization,
            float(kv_stats["utilization"]),
        )
        self.max_waiting = max(self.max_waiting, int(sched_stats["waiting"]))
        self.max_running = max(self.max_running, int(sched_stats["running"]))
        self.max_swapped = max(self.max_swapped, int(sched_stats["swapped"]))


def make_engine(
    *,
    num_gpu_blocks: int,
    block_size: int = 4,
    max_batch_size: int = 4,
    chunk_prefill_size: int = 0,
) -> LLMEngine:
    return LLMEngine(
        EngineConfig(
            model_name="stub",
            dry_run=True,
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
            max_batch_size=max_batch_size,
            chunk_prefill_size=chunk_prefill_size,
            scheduler_policy=SCHEDULER_POLICY,
        )
    )


def drain_engine(
    engine: LLMEngine,
    *,
    stats: RunStats | None = None,
    max_steps: int = 128,
) -> RunStats:
    """Run step() until the engine is idle and collect queue/KV pressure stats."""
    if stats is None:
        stats = RunStats()
    stats.observe(engine)

    while engine.has_unfinished_requests():
        if stats.steps >= max_steps:
            raise RuntimeError(f"scenario did not finish within {max_steps} steps")
        tokens_by_request = engine.step()
        stats.steps += 1
        stats.output_tokens += sum(len(tokens) for tokens in tokens_by_request.values())
        stats.observe(engine)

    return stats


def finish_result(
    name: str,
    engine: LLMEngine,
    *,
    status: str,
    stats: RunStats,
    elapsed_ms: float,
    error: str = "",
) -> ScenarioResult:
    return ScenarioResult(
        name=name,
        status=status,
        steps=stats.steps,
        output_tokens=stats.output_tokens,
        elapsed_ms=elapsed_ms,
        peak_kv_utilization=stats.peak_kv_utilization,
        max_waiting=stats.max_waiting,
        max_running=stats.max_running,
        max_swapped=stats.max_swapped,
        counters=engine.scheduler.stats()["counters"],
        error=error,
    )


def run_scenario(name: str, fn: Callable[[], ScenarioResult]) -> ScenarioResult:
    try:
        return fn()
    except Exception as exc:
        raise RuntimeError(f"scenario {name!r} failed unexpectedly: {exc}") from exc


def scenario_no_pressure() -> ScenarioResult:
    engine = make_engine(num_gpu_blocks=16, max_batch_size=4)
    for prompt in ["aaaa", "bbbb", "cccc"]:
        engine.add_request(prompt, max_new_tokens=4, priority=0)

    t0 = time.perf_counter()
    stats = drain_engine(engine)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return finish_result(
        "no_pressure",
        engine,
        status="ok",
        stats=stats,
        elapsed_ms=elapsed_ms,
    )


def scenario_un_admit_pressure() -> ScenarioResult:
    engine = make_engine(num_gpu_blocks=2, max_batch_size=2)
    # Each request needs ceil((prompt_len=4 + max_new_tokens=4) / block_size=4) = 2 blocks.
    # The second fresh request is admitted then immediately un-admitted because the first
    # request already consumes the whole pool.
    engine.add_request("aaaa", max_new_tokens=4, priority=5)
    engine.add_request("bbbb", max_new_tokens=4, priority=0)

    t0 = time.perf_counter()
    stats = drain_engine(engine)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return finish_result(
        "un_admit_pressure",
        engine,
        status="ok",
        stats=stats,
        elapsed_ms=elapsed_ms,
    )


def scenario_swap_pressure() -> ScenarioResult:
    engine = make_engine(num_gpu_blocks=3, max_batch_size=2)
    # First make the low-priority request prefilled/running. Then add a higher-priority
    # request that needs two blocks. With only one free block left, the scheduler swaps
    # out the low-priority running request.
    engine.add_request("aaaa", max_new_tokens=6, priority=5)

    t0 = time.perf_counter()
    stats = RunStats()
    stats.observe(engine)
    first_tokens = engine.step()
    stats.steps += 1
    stats.output_tokens += sum(len(tokens) for tokens in first_tokens.values())
    stats.observe(engine)

    engine.add_request("bbbb", max_new_tokens=4, priority=0)
    stats.observe(engine)
    drain_engine(engine, stats=stats)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return finish_result(
        "swap_pressure",
        engine,
        status="ok",
        stats=stats,
        elapsed_ms=elapsed_ms,
    )


def scenario_reject_oversize() -> ScenarioResult:
    engine = make_engine(num_gpu_blocks=2, max_batch_size=1)
    # Needs ceil((prompt_len=4 + max_new_tokens=6) / block_size=4) = 3 blocks,
    # which is larger than the whole pool.
    engine.add_request("aaaa", max_new_tokens=6, priority=0)

    t0 = time.perf_counter()
    try:
        drain_engine(engine)
    except RuntimeError as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        stats = RunStats()
        stats.observe(engine)
        return finish_result(
            "reject_oversize",
            engine,
            status="rejected",
            stats=stats,
            elapsed_ms=elapsed_ms,
            error=str(exc),
        )
    raise RuntimeError("reject_oversize should have been rejected")


def workload_short_long_mix() -> ScenarioResult:
    engine = make_engine(num_gpu_blocks=4, max_batch_size=4)
    # Long request: prompt_len=12, max_new_tokens=4, needs all 4 blocks.
    # Later short requests have the same priority, so the current scheduler will not
    # preempt the long request; they wait until the long request finishes.
    engine.add_request("L" * 12, max_new_tokens=4, priority=0)

    t0 = time.perf_counter()
    stats = RunStats()
    stats.observe(engine)
    first_tokens = engine.step()
    stats.steps += 1
    stats.output_tokens += sum(len(tokens) for tokens in first_tokens.values())
    stats.observe(engine)

    for prompt in ["a", "b", "c"]:
        engine.add_request(prompt, max_new_tokens=2, priority=0)
    stats.observe(engine)

    drain_engine(engine, stats=stats)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return finish_result(
        "workload_short_long_mix",
        engine,
        status="ok",
        stats=stats,
        elapsed_ms=elapsed_ms,
    )


def workload_priority_late_arrival() -> ScenarioResult:
    engine = make_engine(num_gpu_blocks=3, max_batch_size=2)
    # Low-priority request starts first and occupies 2 blocks. A high-priority request
    # arrives later and needs 2 blocks, so the scheduler swaps out the low-priority one.
    engine.add_request("low0", max_new_tokens=6, priority=5)

    t0 = time.perf_counter()
    stats = RunStats()
    stats.observe(engine)
    first_tokens = engine.step()
    stats.steps += 1
    stats.output_tokens += sum(len(tokens) for tokens in first_tokens.values())
    stats.observe(engine)

    engine.add_request("high", max_new_tokens=4, priority=0)
    stats.observe(engine)

    drain_engine(engine, stats=stats)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return finish_result(
        "workload_priority_late_arrival",
        engine,
        status="ok",
        stats=stats,
        elapsed_ms=elapsed_ms,
    )


def workload_kv_oversubscription() -> ScenarioResult:
    engine = make_engine(num_gpu_blocks=3, max_batch_size=4)
    # Each request needs 2 blocks, but the pool has only 3 blocks. Total demand is
    # 10 blocks. The current scheduler can over-admit because init_request only
    # allocates prompt blocks; later decode growth can hit KV OOM. This is a useful
    # baseline failure for pressure-aware admission.
    for prompt in ["aa", "bb", "cc", "dd", "ee"]:
        engine.add_request(prompt, max_new_tokens=5, priority=0)

    t0 = time.perf_counter()
    stats = RunStats()
    try:
        drain_engine(engine, stats=stats)
    except RuntimeError as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return finish_result(
            "workload_kv_oversubscription",
            engine,
            status="decode_oom",
            stats=stats,
            elapsed_ms=elapsed_ms,
            error=str(exc),
        )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return finish_result(
        "workload_kv_oversubscription",
        engine,
        status="ok",
        stats=stats,
        elapsed_ms=elapsed_ms,
    )


def print_table(results: list[ScenarioResult]) -> None:
    headers = [
        "scenario",
        "status",
        "steps",
        "tokens",
        "elapsed_ms",
        "peak_kv",
        "max_wait",
        "max_run",
        "max_swap",
        "admit",
        "reject",
        "preempt",
        "un_admit",
        "swap_out",
        "swap_in",
        "finish",
    ]
    rows = []
    for result in results:
        c = result.counters
        rows.append([
            result.name,
            result.status,
            str(result.steps),
            str(result.output_tokens),
            f"{result.elapsed_ms:.2f}",
            f"{result.peak_kv_utilization:.2f}",
            str(result.max_waiting),
            str(result.max_running),
            str(result.max_swapped),
            str(c["admit_count"]),
            str(c["reject_count"]),
            str(c["preempt_count"]),
            str(c["un_admit_count"]),
            str(c["swap_out_count"]),
            str(c["swap_in_count"]),
            str(c["finish_count"]),
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
    parser = argparse.ArgumentParser(description="Dry-run scheduler counter benchmark")
    parser.add_argument(
        "--json",
        action="store_true",
        help="print JSON instead of a human-readable table",
    )
    parser.add_argument(
        "--suite",
        choices=["micro", "workload", "all"],
        default="all",
        help="which scenario suite to run",
    )
    parser.add_argument(
        "--scheduler-policy",
        choices=["baseline", "reserve_only", "pressure_aware", "adaptive", "slo_aware"],
        default="pressure_aware",
    )
    return parser.parse_args()


def main() -> None:
    global SCHEDULER_POLICY
    args = parse_args()
    SCHEDULER_POLICY = args.scheduler_policy
    micro_scenarios = [
        ("no_pressure", scenario_no_pressure),
        ("un_admit_pressure", scenario_un_admit_pressure),
        ("swap_pressure", scenario_swap_pressure),
        ("reject_oversize", scenario_reject_oversize),
    ]
    workload_scenarios = [
        ("workload_short_long_mix", workload_short_long_mix),
        ("workload_priority_late_arrival", workload_priority_late_arrival),
        ("workload_kv_oversubscription", workload_kv_oversubscription),
    ]
    if args.suite == "micro":
        scenarios = micro_scenarios
    elif args.suite == "workload":
        scenarios = workload_scenarios
    else:
        scenarios = micro_scenarios + workload_scenarios
    results = [run_scenario(name, fn) for name, fn in scenarios]

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        print_table(results)


if __name__ == "__main__":
    main()
