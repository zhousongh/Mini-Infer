# Scheduler Policy Layer

This layer turns the serving runtime from "can batch requests" into "can make pressure-aware admission and preemption decisions".

## Policies

| Policy | Meaning |
| --- | --- |
| `baseline` | Greedy FIFO admission. No decode KV reservation. Same-priority requests do not preempt. |
| `reserve_only` | Reserves KV blocks for prompt + future decode before admission. Prevents decode-time KV OOM, but can serialize too much. |
| `pressure_aware` | KV reservation plus pressure-aware preemption. Lower-priority or much larger same-priority requests can be preempted. |
| `adaptive` | Uses baseline behavior under low pressure, then enables reservation/preemption as projected KV utilization rises. |
| `slo_aware` | KV reservation plus pressure-aware preemption plus SLO-aware waiting selection. Waiting requests are ranked by priority, TTFT SLO risk, waiting age, and size. |

## SLO-Aware Scope

`slo_aware` is intentionally conservative in this first version.

It adds:

1. Priority-aware waiting selection instead of pure FIFO.
2. Age-aware selection for requests at TTFT SLO risk.
3. Short-job preference under the same priority and SLO-risk level.
4. Same-priority preemption only when the victim can free enough KV blocks and is not close to finish.

It does not try to maximize TTFT alone. TTFT is one service-quality signal, not the whole service-quality objective. The policy also tracks safety under KV pressure, completion rate, latency, throughput, preemption churn, and starvation risk.

## Commands

Compare all policies on a synthetic pressure trace:

```bash
PYTHONPATH=. python benchmarks/benchmark_scheduler_trace.py \
  --num-gpu-blocks 24 \
  --compare-policies
```

Run the hand-written pressure workloads:

```bash
PYTHONPATH=. python benchmarks/benchmark_scheduler_counters.py \
  --suite workload \
  --scheduler-policy slo_aware
```

Run the HTTP server with `slo_aware`:

```bash
mini-infer-serve \
  --model /home/zsh/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct \
  --device cuda:0 \
  --port 8000 \
  --scheduler-policy slo_aware \
  --scheduler-ttft-slo-steps 16
```

## Local Results

Synthetic pressure trace on 2026-06-23, `num_gpu_blocks=24`:

| Policy | Status | Completed | Steps | Tok/step | P95 TTFT | TTFT miss | P95 latency | Lat miss | Preempt |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `decode_oom` | 41/50 | 72 | 6.56 | 26 | 11% | 33 | 0% | 0 |
| `reserve_only` | `ok` | 50/50 | 155 | 3.95 | 75 | 22% | 81 | 14% | 0 |
| `pressure_aware` | `ok` | 50/50 | 125 | 4.90 | 30 | 22% | 50 | 4% | 2 |
| `adaptive` | `ok` | 50/50 | 125 | 4.90 | 30 | 22% | 50 | 4% | 2 |
| `slo_aware` | `ok` | 50/50 | 117 | 5.24 | 17 | 8% | 43 | 2% | 1 |

Interpretation:

- `baseline` is fastest before it fails, but it is not safe under KV pressure.
- `reserve_only` is safe but overly conservative.
- `pressure_aware` is the current best default for this synthetic trace.
- `slo_aware` is safe on this trace and improves P95 TTFT and TTFT miss rate by prioritizing short jobs within the same priority/SLO-risk class.
- `slo_aware` should still be evaluated as a multi-objective serving policy, not as a TTFT-only optimizer.

Workload suite with `slo_aware`:

| Scenario | Status | Steps | Peak KV | Max waiting | Preempt | Swap out/in | Finish |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `workload_short_long_mix` | `ok` | 4 | 1.00 | 3 | 1 | 1/1 | 4 |
| `workload_priority_late_arrival` | `ok` | 8 | 0.67 | 1 | 1 | 1/1 | 2 |
| `workload_kv_oversubscription` | `ok` | 20 | 0.67 | 5 | 3 | 0/0 | 5 |

## Next Scheduler Work

1. Add trace comparison focused on request classes: urgent, short, medium, long.
2. Tune `slo_aware` scoring weights so it improves TTFT SLO miss rate without hurting throughput too much.
3. Add a real HTTP workload that mixes long prompt prefill with short streaming requests.
4. Decide whether `slo_aware` should be a standalone policy or folded into `adaptive`.
