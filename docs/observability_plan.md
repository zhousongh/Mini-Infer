# Observability Plan

This layer makes the serving runtime explainable before adding more scheduler policy.

## Goal

For each request, answer:

1. When was it added?
2. When was it admitted by the scheduler?
3. When did prefill finish?
4. When did it produce the first token?
5. When did it finish?
6. How many KV blocks and queue slots were used while it was active?
7. Was it preempted, swapped, rejected, or canceled?

For each workload, answer:

1. Peak and average KV utilization.
2. Waiting/running/prefilling/swapped queue pressure.
3. TTFT and latency p95.
4. Scheduler counters: admit, reject, preempt, swap out, swap in, finish.

## Current State

Already available:

- `/healthz` returns KV cache and scheduler snapshots.
- `MINI_INFER_KV_TRACE_FILE=/path/to/trace.jsonl` records per-step KV and scheduler state.
- Scheduler counters track admit/reject/preempt/swap/finish events.
- Synthetic scheduler benchmarks compare baseline, reserve-only, pressure-aware, and adaptive policies.

Added:

- `benchmarks/analyze_kv_trace.py` turns raw JSONL into request lifecycle metrics and global pressure metrics.
- `benchmarks/benchmark_http_streaming.py` measures true HTTP streaming TTFT, ITL, and latency over a real socket.
- `tests/test_trace_analyzer.py` covers trace parsing and lifecycle recovery.
- `tests/test_http_streaming_benchmark.py` covers SSE parsing helpers.

Example:

```bash
MINI_INFER_KV_TRACE_FILE=/tmp/kv_trace.jsonl \
PYTHONPATH=. python benchmarks/benchmark_scheduler_trace.py \
  --num-requests 50 --num-gpu-blocks 24 --scheduler-policy pressure_aware

PYTHONPATH=. python benchmarks/analyze_kv_trace.py /tmp/kv_trace.jsonl
PYTHONPATH=. python benchmarks/analyze_kv_trace.py /tmp/kv_trace.jsonl --json
```

Real HTTP streaming benchmark:

```bash
# against an already running server
PYTHONPATH=. python benchmarks/benchmark_http_streaming.py \
  --url http://127.0.0.1:8000 \
  --concurrency 1 2 4 8 \
  --max-tokens 64 \
  --tokenizer /home/zsh/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct

# or launch a temporary dry-run server
PYTHONPATH=. python benchmarks/benchmark_http_streaming.py \
  --dry-run \
  --host 127.0.0.1 \
  --port 18080 \
  --concurrency 1 2 4 \
  --max-tokens 8

# or launch a temporary real-model server
CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=. python benchmarks/benchmark_http_streaming.py \
  --model /home/zsh/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct \
  --host 127.0.0.1 \
  --port 18080 \
  --device cuda:0 \
  --num-gpu-blocks 256 \
  --concurrency 1 2 4 8 \
  --max-tokens 64 \
  --tokenizer /home/zsh/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct
```

Local smoke test on 2026-06-23 with Qwen2.5-7B, physical GPU 1, `max_tokens=16`:

| Concurrency | TTFT mean | Latency mean | ITL mean | Token throughput |
| ---: | ---: | ---: | ---: | ---: |
| 1 | `58.3 ms` | `416.4 ms` | `25.5 ms` | `38.4 tok/s` |
| 2 | `87.1 ms` | `454.4 ms` | `26.2 ms` | `70.1 tok/s` |

## Next Steps

1. Add per-request exported metrics in `AsyncEngine`, so `/healthz` or a future `/metrics` endpoint can expose recent completed request stats.
2. Add trace comparison tooling for two policies over the same workload.
3. Use these metrics as the evaluation harness for the next scheduler changes.
