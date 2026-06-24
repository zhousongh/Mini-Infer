# mini-infer

mini-infer 是一个从零实现的 LLM 推理引擎原型，覆盖 Paged KV Cache、Continuous Batching、Chunked Prefill、Prefix Caching、True PagedAttention、OpenAI-compatible HTTP Server，以及若干独立推理优化 benchmark。

本分支在原项目基础上进一步补充了三类工作：

1. **主线 serving path 复现**：复现 Qwen2.5-7B 下的 HF baseline、True PagedAttention、HTTP continuous batching、Chunked Prefill、Prefix Caching、Flash Decoding 等结果。
2. **可观测性建设**：新增 `/healthz` KV/Scheduler 快照、JSONL request lifecycle trace、离线 trace analyzer、真实 HTTP streaming benchmark，用于测量 TTFT、ITL、P95 latency、KV utilization 和调度事件。
3. **调度策略扩展**：实现 KV pressure-aware 与 SLO-aware Scheduler，支持 KV block reservation、priority/short-job/age-aware admission 与保守 preemption，并提供 synthetic pressure trace 对比。

---

## 项目定位

mini-infer 的重点不是“再封装一个 `model.generate()`”，而是把 LLM serving runtime 里的核心机制拆开实现和测量：

```text
HTTP API
  -> AsyncEngine
  -> Scheduler
  -> Prefill / Decode
  -> KVCacheManager
  -> PagedAttention
  -> Token Streaming
```

它适合用来学习和实验：

- 请求生命周期：从 `/v1/chat/completions` 到 token 返回。
- KV block 分配、释放、预留、换出和前缀复用。
- Continuous batching 下 waiting / running / prefilling / swapped 队列协作。
- Prefill 与 decode 的延迟、吞吐和调度权衡。
- KV 压力下的 admission / preemption / rejection 策略。

---

## 快速启动

### 无权重 dry-run 验证

```bash
pip install -e ".[serve,dev]"
mini-infer-serve --dry-run --port 8000
curl --noproxy '*' http://127.0.0.1:8000/healthz
```

### 使用本地 Qwen2.5-7B 权重

```bash
CUDA_VISIBLE_DEVICES=1 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
mini-infer-serve \
  --model /home/zsh/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda:0 \
  --dtype float16 \
  --num-gpu-blocks 256
```

### OpenAI-compatible 请求

```bash
curl --noproxy '*' http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mini-infer",
    "messages": [{"role": "user", "content": "你好，简单介绍一下你自己"}],
    "stream": false,
    "max_tokens": 128
  }'
```

---

## 主线 serving path 复现结果

本地环境：NVIDIA RTX A6000，Qwen2.5-7B-Instruct，`float16`。

| 项目 | 本地结果 | 说明 |
| --- | ---: | --- |
| HF Transformers baseline | 296.8 tok/s | `benchmark_hf.py`，batch=8，max_new_tokens=128 |
| mini-infer True PagedAttention | 295.0 tok/s | 达到 HF baseline 99.4% |
| HTTP continuous batching | 39.9 -> 226.8 tok/s | 并发 1 -> 8 |
| Chunked Prefill, chunk=256 | ITL spike -61.8% | 长短请求混合场景 |
| Chunked Prefill, chunk=128 | ITL spike -67.0% | 更小 chunk 换更平滑 decode |
| Prefix Caching | 单请求 1.47x 加速 | 共享前缀命中 |
| Flash Decoding, seq=4096 | 3.28x | 相对 Phase 6.5 Triton decode kernel |

复现命令集中记录在：

- [docs/reproduce_readme_results.md](docs/reproduce_readme_results.md)
- [docs/benchmarks.md](docs/benchmarks.md)

---

## 可观测性层

### `/healthz` 状态快照

`/healthz` 会返回当前模型、调度策略、KV cache 和 scheduler 状态：

```bash
curl --noproxy '*' http://127.0.0.1:8000/healthz
```

核心字段包括：

- KV cache：`total_blocks`、`free_blocks`、`used_blocks`、`utilization`
- Reservation：`reserved_blocks`、`outstanding_reserved_blocks`
- Scheduler queue：`waiting`、`prefilling`、`running`、`swapped`
- Scheduler counters：`admit_count`、`reject_count`、`preempt_count`、`swap_out_count`、`swap_in_count`、`finish_count`

### JSONL request lifecycle trace

设置环境变量后，每个 engine step 会记录 KV/Scheduler 快照：

```bash
MINI_INFER_KV_TRACE_FILE=/tmp/kv_trace.jsonl \
PYTHONPATH=. python benchmarks/benchmark_scheduler_trace.py \
  --num-requests 50 \
  --num-gpu-blocks 24 \
  --scheduler-policy pressure_aware
```

离线分析：

```bash
PYTHONPATH=. python benchmarks/analyze_kv_trace.py /tmp/kv_trace.jsonl
PYTHONPATH=. python benchmarks/analyze_kv_trace.py /tmp/kv_trace.jsonl --json
```

可恢复的生命周期指标：

- `add_request`
- `admit`
- `prefill_complete`
- `first_decode`
- `finish_request`
- TTFT
- latency
- KV utilization
- queue pressure
- preempt / reject / swap 计数

### 真实 HTTP streaming benchmark

原 `benchmark_server.py` 使用 `ASGITransport`，会缓冲响应，不能测真实 TTFT。本分支新增真实 HTTP streaming benchmark，通过 socket 消费 SSE 流：

```bash
PYTHONPATH=. python benchmarks/benchmark_http_streaming.py \
  --url http://127.0.0.1:8000 \
  --concurrency 1 2 4 8 \
  --max-tokens 64 \
  --tokenizer /home/zsh/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct
```

小规模真实模型 smoke test：

| Concurrency | TTFT mean | Latency mean | ITL mean | Throughput |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 58.3 ms | 416.4 ms | 25.5 ms | 38.4 tok/s |
| 2 | 87.1 ms | 454.4 ms | 26.2 ms | 70.1 tok/s |

更多说明见 [docs/observability_plan.md](docs/observability_plan.md)。

---

## KV pressure-aware / SLO-aware Scheduler

本分支新增和对比了多种调度策略：

| Policy | 说明 |
| --- | --- |
| `baseline` | FIFO 贪心准入；不做 decode KV reservation；同优先级不抢占 |
| `reserve_only` | 准入时预留 prompt + future decode 所需 KV blocks，避免 decode-time OOM |
| `pressure_aware` | KV reservation + pressure-aware preemption，允许低优先级或明显更大的同优先级请求被抢占 |
| `adaptive` | 低压走 baseline，中高压启用 reservation / pressure preemption |
| `slo_aware` | 在 KV pressure-aware 基础上加入 priority、SLO risk、short-job、waiting age 的 waiting selection |

### 启动服务时选择策略

```bash
mini-infer-serve \
  --model /home/zsh/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct \
  --device cuda:0 \
  --port 8000 \
  --scheduler-policy slo_aware \
  --scheduler-ttft-slo-steps 16
```

### Synthetic pressure trace 结果

```bash
PYTHONPATH=. python benchmarks/benchmark_scheduler_trace.py \
  --num-gpu-blocks 24 \
  --compare-policies
```

| Policy | Status | Completed | Steps | Tok/step | P95 TTFT | TTFT miss | P95 latency | Lat miss | Preempt |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline` | `decode_oom` | 41/50 | 72 | 6.56 | 26 | 11% | 33 | 0% | 0 |
| `reserve_only` | `ok` | 50/50 | 155 | 3.95 | 75 | 22% | 81 | 14% | 0 |
| `pressure_aware` | `ok` | 50/50 | 125 | 4.90 | 30 | 22% | 50 | 4% | 2 |
| `adaptive` | `ok` | 50/50 | 125 | 4.90 | 30 | 22% | 50 | 4% | 2 |
| `slo_aware` | `ok` | 50/50 | 117 | 5.24 | 17 | 8% | 43 | 2% | 1 |

结论：

- `baseline` 吞吐高但不安全，KV 压力下会 decode OOM。
- `reserve_only` 安全但过于保守。
- `pressure_aware` 在压力场景下能避免 OOM，并保持较好的 latency。
- `slo_aware` 在本 synthetic trace 中进一步降低 P95 TTFT 和 TTFT miss rate，但应作为多目标 serving policy 评估，不应只看 TTFT 一个数字。

更多说明见 [docs/scheduler_policy_plan.md](docs/scheduler_policy_plan.md)。

---

## 原项目核心 benchmark

原 README 中的主要结果如下，完整命令见 [docs/benchmarks.md](docs/benchmarks.md)。

**主 serving 路径**

| 技术 | 原始结果 |
| --- | --- |
| Continuous Batching HTTP API | 并发 1 -> 8 吞吐 55.7 -> 219.1 tok/s |
| True PagedAttention | batch=8 达到 HF Transformers 100% 吞吐，约 406 tok/s |
| Chunked Prefill | 混合 serving 场景 ITL spike 降低 57% - 67% |
| Prefix Caching | 共享前缀 TTFT 降低约 22% |

**独立实验**

| 技术 | 原始结果 |
| --- | --- |
| Speculative Decoding | acceptance rate 55.85% |
| CUDA Graph | 1.5B bs=1 decode latency 降低 28.9% |
| Flash Decoding | seq=4096 相对标准 Triton 约 3.31x |
| Tensor Parallelism | TP=2 greedy 输出与单卡完全一致 |
| PD Disaggregation | prefill 12.3ms / transfer 约 14.7ms / decode 519ms |

本地权重状态会影响复现范围：当前本地有 Qwen2.5-7B/14B/32B，缺 Qwen2.5-0.5B、Qwen2.5-1.5B、DeepSeek-V2-Lite，因此 speculative decoding、CUDA Graph 原始条件、TP、quant、PD、MLA 的完整复现需要补权重。

---

## 目录结构

```text
mini_infer/
├─ core/        # EngineConfig、Request、SamplingParams
├─ runtime/     # LLMEngine、Scheduler、AsyncEngine、SpecEngine、PDEngine
├─ cache/       # KVCacheManager、BlockTable、Prefix Cache
├─ modeling/    # ModelRunner
├─ kernels/     # PagedAttention、Triton decode、Flash Decoding
├─ parallel/    # TP、Replica、PP
└─ serving/     # FastAPI server、OpenAI schema

benchmarks/
├─ benchmark_server.py             # ASGITransport HTTP benchmark
├─ benchmark_http_streaming.py     # 真实 HTTP streaming TTFT/ITL benchmark
├─ benchmark_scheduler_trace.py    # synthetic scheduler trace
├─ benchmark_scheduler_counters.py # 手写压力 workload
└─ analyze_kv_trace.py             # JSONL trace 离线分析

docs/
├─ reproduce_readme_results.md
├─ observability_plan.md
└─ scheduler_policy_plan.md
```

---

## 测试

常用轻量测试：

```bash
PYTHONPATH=. pytest \
  tests/test_kv_cache.py \
  tests/test_preemption.py \
  tests/test_engine.py \
  tests/test_server.py \
  tests/test_trace_analyzer.py \
  tests/test_http_streaming_benchmark.py \
  -q
```

当前相关测试通过：

```text
62 passed
```

---

## 与 vLLM 的区别

| 维度 | mini-infer | vLLM |
| --- | --- | --- |
| 目标 | 教学/研究型推理引擎，强调机制清晰和 benchmark 可复现 | 生产级 LLM serving 框架 |
| PagedAttention | 与 vLLM 同路线，使用 block table | 更成熟、更完整 |
| 调度器 | 手工实现，便于实验 KV pressure / SLO-aware 策略 | 更完整的生产调度和多租户能力 |
| 可观测性 | 本分支新增 healthz、JSONL trace、streaming benchmark | 生产级 metrics / tracing / deployment 生态 |
| 模型覆盖 | 主要围绕 Qwen2.5 / DeepSeek-V2 相关实验 | 覆盖大量模型架构 |
| 部署 | 单机原型为主 | 多机、多副本、生产部署 |

mini-infer 的价值不在于替代 vLLM，而在于把 LLM serving runtime 的核心机制拆开、实现、测量，并允许快速实验调度策略。

---

## 环境

| 依赖 | 版本 |
| --- | --- |
| Python | 3.10+ |
| PyTorch | 2.1.2+cu121 |
| transformers | 4.43.4 |
| accelerate | 1.14.0 |
| flash-attn | 2.5.9.post1 |
| CUDA | 12.1 |

---

## License

MIT
