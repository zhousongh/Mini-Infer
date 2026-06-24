"""Analyze MINI_INFER_KV_TRACE_FILE JSONL traces.

The engine trace is intentionally low level: every event stores a scheduler
snapshot and a KV cache snapshot. This script turns that stream into request
lifecycle metrics and global pressure metrics that are easier to compare across
scheduler policies.

Example:
  MINI_INFER_KV_TRACE_FILE=/tmp/kv_trace.jsonl python benchmarks/benchmark_scheduler_trace.py
  python benchmarks/analyze_kv_trace.py /tmp/kv_trace.jsonl
  python benchmarks/analyze_kv_trace.py /tmp/kv_trace.jsonl --json
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RequestLifecycle:
    request_id: str
    add_step: int | None = None
    add_ts: float | None = None
    prompt_len: int | None = None
    max_new_tokens: int | None = None
    priority: int | None = None
    admit_step: int | None = None
    prefill_done_step: int | None = None
    first_decode_step: int | None = None
    first_decode_ts: float | None = None
    finish_step: int | None = None
    finish_ts: float | None = None
    finish_reason: str | None = None
    generated_tokens: int | None = None
    cancel_step: int | None = None
    swap_out_count: int = 0
    swap_in_count: int = 0
    preempt_count: int = 0
    decode_steps: int = 0

    @property
    def status(self) -> str:
        if self.finish_step is not None:
            return "finished"
        if self.cancel_step is not None:
            return "canceled"
        return "unfinished"

    @property
    def ttft_steps(self) -> int | None:
        if self.add_step is None or self.first_decode_step is None:
            return None
        return self.first_decode_step - self.add_step + 1

    @property
    def latency_steps(self) -> int | None:
        if self.add_step is None or self.finish_step is None:
            return None
        return self.finish_step - self.add_step + 1

    @property
    def ttft_ms(self) -> float | None:
        if self.add_ts is None or self.first_decode_ts is None:
            return None
        return (self.first_decode_ts - self.add_ts) * 1000

    @property
    def latency_ms(self) -> float | None:
        if self.add_ts is None or self.finish_ts is None:
            return None
        return (self.finish_ts - self.add_ts) * 1000


@dataclass
class TraceAnalysis:
    path: str
    records: int
    steps: int
    wall_ms: float
    requests: list[RequestLifecycle]
    completed: int
    canceled: int
    unfinished: int
    mean_ttft_steps: float
    p95_ttft_steps: float
    mean_latency_steps: float
    p95_latency_steps: float
    mean_ttft_ms: float
    p95_ttft_ms: float
    mean_latency_ms: float
    p95_latency_ms: float
    peak_kv_utilization: float
    avg_step_kv_utilization: float
    peak_used_blocks: int
    peak_reserved_blocks: int
    max_waiting: int
    max_prefilling: int
    max_running: int
    max_swapped: int
    scheduler_counters: dict[str, int] = field(default_factory=dict)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int((len(ordered) - 1) * pct))
    return float(ordered[idx])


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: expected JSON object")
            records.append(record)
    return records


def as_request_ids(record: dict[str, Any]) -> list[str]:
    if "request_id" in record and record["request_id"] is not None:
        return [str(record["request_id"])]
    if "request_ids" in record and record["request_ids"] is not None:
        return [str(rid) for rid in record["request_ids"]]
    return []


def get_lifecycle(lifecycles: dict[str, RequestLifecycle], request_id: str) -> RequestLifecycle:
    lifecycle = lifecycles.get(request_id)
    if lifecycle is None:
        lifecycle = RequestLifecycle(request_id=request_id)
        lifecycles[request_id] = lifecycle
    return lifecycle


def analyze(path: Path) -> TraceAnalysis:
    records = read_records(path)
    lifecycles: dict[str, RequestLifecycle] = {}
    step_kv_utilization: dict[int, float] = {}
    first_ts = records[0]["ts"] if records else 0.0
    last_ts = records[-1]["ts"] if records else first_ts
    peak_kv_utilization = 0.0
    peak_used_blocks = 0
    peak_reserved_blocks = 0
    max_waiting = 0
    max_prefilling = 0
    max_running = 0
    max_swapped = 0
    scheduler_counters: dict[str, int] = {}

    for record in records:
        event = str(record.get("event", ""))
        step = int(record.get("step", 0))
        ts = float(record.get("ts", 0.0))

        kv = record.get("kv_cache") or {}
        if isinstance(kv, dict):
            utilization = float(kv.get("utilization", 0.0))
            step_kv_utilization[step] = max(
                step_kv_utilization.get(step, 0.0),
                utilization,
            )
            peak_kv_utilization = max(peak_kv_utilization, utilization)
            peak_used_blocks = max(peak_used_blocks, int(kv.get("used_blocks", 0)))
            peak_reserved_blocks = max(
                peak_reserved_blocks,
                int(kv.get("reserved_blocks", 0)),
            )

        scheduler = record.get("scheduler") or {}
        if isinstance(scheduler, dict):
            max_waiting = max(max_waiting, int(scheduler.get("waiting", 0)))
            max_prefilling = max(max_prefilling, int(scheduler.get("prefilling", 0)))
            max_running = max(max_running, int(scheduler.get("running", 0)))
            max_swapped = max(max_swapped, int(scheduler.get("swapped", 0)))
            counters = scheduler.get("counters")
            if isinstance(counters, dict):
                scheduler_counters = {str(k): int(v) for k, v in counters.items()}

        if event in {"add_request", "add_request_generate"}:
            for request_id in as_request_ids(record):
                lifecycle = get_lifecycle(lifecycles, request_id)
                lifecycle.add_step = step
                lifecycle.add_ts = ts
                lifecycle.prompt_len = int(record.get("prompt_len", 0))
                lifecycle.max_new_tokens = int(record.get("max_new_tokens", 0))
                lifecycle.priority = int(record.get("priority", 0))

        elif event.startswith("admit_"):
            for request_id in as_request_ids(record):
                lifecycle = get_lifecycle(lifecycles, request_id)
                if lifecycle.admit_step is None:
                    lifecycle.admit_step = step

        elif event == "prefill_complete":
            for request_id in as_request_ids(record):
                lifecycle = get_lifecycle(lifecycles, request_id)
                if lifecycle.prefill_done_step is None:
                    lifecycle.prefill_done_step = step

        elif event == "decode_batch":
            for request_id in as_request_ids(record):
                lifecycle = get_lifecycle(lifecycles, request_id)
                lifecycle.decode_steps += 1
                if lifecycle.first_decode_step is None:
                    lifecycle.first_decode_step = step
                    lifecycle.first_decode_ts = ts

        elif event == "finish_request":
            for request_id in as_request_ids(record):
                lifecycle = get_lifecycle(lifecycles, request_id)
                lifecycle.finish_step = step
                lifecycle.finish_ts = ts
                lifecycle.finish_reason = (
                    str(record.get("finish_reason"))
                    if record.get("finish_reason") is not None
                    else None
                )
                lifecycle.generated_tokens = int(record.get("generated_tokens", 0))

        elif event == "cancel_request":
            for request_id in as_request_ids(record):
                lifecycle = get_lifecycle(lifecycles, request_id)
                lifecycle.cancel_step = step

        elif event in {"swap_out", "preempt"}:
            for request_id in as_request_ids(record):
                lifecycle = get_lifecycle(lifecycles, request_id)
                lifecycle.swap_out_count += 1
                lifecycle.preempt_count += 1

        elif event == "swap_in":
            for request_id in as_request_ids(record):
                lifecycle = get_lifecycle(lifecycles, request_id)
                lifecycle.swap_in_count += 1

    requests = sorted(
        lifecycles.values(),
        key=lambda req: (
            req.add_step if req.add_step is not None else 10**12,
            req.request_id,
        ),
    )
    completed_requests = [req for req in requests if req.status == "finished"]
    canceled_requests = [req for req in requests if req.status == "canceled"]
    unfinished_requests = [req for req in requests if req.status == "unfinished"]

    ttft_steps = [req.ttft_steps for req in requests if req.ttft_steps is not None]
    latency_steps = [
        req.latency_steps for req in completed_requests
        if req.latency_steps is not None
    ]
    ttft_ms = [req.ttft_ms for req in requests if req.ttft_ms is not None]
    latency_ms = [
        req.latency_ms for req in completed_requests
        if req.latency_ms is not None
    ]

    max_step = max((int(record.get("step", 0)) for record in records), default=0)
    return TraceAnalysis(
        path=str(path),
        records=len(records),
        steps=max_step,
        wall_ms=(float(last_ts) - float(first_ts)) * 1000,
        requests=requests,
        completed=len(completed_requests),
        canceled=len(canceled_requests),
        unfinished=len(unfinished_requests),
        mean_ttft_steps=statistics.mean(ttft_steps) if ttft_steps else 0.0,
        p95_ttft_steps=percentile([float(v) for v in ttft_steps], 0.95),
        mean_latency_steps=statistics.mean(latency_steps) if latency_steps else 0.0,
        p95_latency_steps=percentile([float(v) for v in latency_steps], 0.95),
        mean_ttft_ms=statistics.mean(ttft_ms) if ttft_ms else 0.0,
        p95_ttft_ms=percentile(ttft_ms, 0.95),
        mean_latency_ms=statistics.mean(latency_ms) if latency_ms else 0.0,
        p95_latency_ms=percentile(latency_ms, 0.95),
        peak_kv_utilization=peak_kv_utilization,
        avg_step_kv_utilization=(
            statistics.mean(step_kv_utilization.values())
            if step_kv_utilization else 0.0
        ),
        peak_used_blocks=peak_used_blocks,
        peak_reserved_blocks=peak_reserved_blocks,
        max_waiting=max_waiting,
        max_prefilling=max_prefilling,
        max_running=max_running,
        max_swapped=max_swapped,
        scheduler_counters=scheduler_counters,
    )


def request_to_row(req: RequestLifecycle) -> list[str]:
    def fmt(value: object) -> str:
        return "-" if value is None else str(value)

    return [
        req.request_id,
        req.status,
        fmt(req.priority),
        fmt(req.prompt_len),
        fmt(req.max_new_tokens),
        fmt(req.add_step),
        fmt(req.admit_step),
        fmt(req.first_decode_step),
        fmt(req.finish_step),
        fmt(req.ttft_steps),
        fmt(req.latency_steps),
        fmt(req.generated_tokens),
        str(req.preempt_count),
        str(req.swap_in_count),
    ]


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        if rows else len(headers[i])
        for i in range(len(headers))
    ]
    print("  " + "  ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("  " + "  ".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print("  " + "  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def print_human(analysis: TraceAnalysis, max_requests: int) -> None:
    print("KV trace analysis")
    print(f"  file                 : {analysis.path}")
    print(f"  records / steps      : {analysis.records} / {analysis.steps}")
    print(f"  trace wall time      : {analysis.wall_ms:.1f} ms")
    print(f"  requests             : {len(analysis.requests)}")
    print(f"  completed/canceled/un: {analysis.completed}/{analysis.canceled}/{analysis.unfinished}")
    print(f"  TTFT steps           : mean={analysis.mean_ttft_steps:.2f} p95={analysis.p95_ttft_steps:.2f}")
    print(f"  latency steps        : mean={analysis.mean_latency_steps:.2f} p95={analysis.p95_latency_steps:.2f}")
    print(f"  TTFT ms              : mean={analysis.mean_ttft_ms:.1f} p95={analysis.p95_ttft_ms:.1f}")
    print(f"  latency ms           : mean={analysis.mean_latency_ms:.1f} p95={analysis.p95_latency_ms:.1f}")
    print(f"  KV utilization       : peak={analysis.peak_kv_utilization:.2f} avg_step={analysis.avg_step_kv_utilization:.2f}")
    print(f"  KV blocks            : peak_used={analysis.peak_used_blocks} peak_reserved={analysis.peak_reserved_blocks}")
    print(f"  queues max           : waiting={analysis.max_waiting} prefilling={analysis.max_prefilling} running={analysis.max_running} swapped={analysis.max_swapped}")
    if analysis.scheduler_counters:
        counters = ", ".join(
            f"{key}={value}"
            for key, value in sorted(analysis.scheduler_counters.items())
        )
        print(f"  scheduler counters   : {counters}")

    print("\nRequest lifecycle")
    rows = [request_to_row(req) for req in analysis.requests[:max_requests]]
    print_table(
        [
            "request_id", "status", "prio", "prompt", "max_new",
            "add", "admit", "first", "finish", "ttft", "latency",
            "gen", "preempt", "swap_in",
        ],
        rows,
    )
    if len(analysis.requests) > max_requests:
        print(f"  ... {len(analysis.requests) - max_requests} more requests omitted")


def analysis_to_jsonable(analysis: TraceAnalysis) -> dict[str, Any]:
    data = asdict(analysis)
    data["requests"] = [
        {
            **asdict(req),
            "status": req.status,
            "ttft_steps": req.ttft_steps,
            "latency_steps": req.latency_steps,
            "ttft_ms": req.ttft_ms,
            "latency_ms": req.latency_ms,
        }
        for req in analysis.requests
    ]
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze mini-infer KV trace JSONL")
    parser.add_argument("trace_file", type=Path)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--max-requests", type=int, default=20)
    args = parser.parse_args()

    analysis = analyze(args.trace_file)
    if args.json:
        print(json.dumps(analysis_to_jsonable(analysis), ensure_ascii=False, indent=2))
    else:
        print_human(analysis, max_requests=args.max_requests)


if __name__ == "__main__":
    main()
