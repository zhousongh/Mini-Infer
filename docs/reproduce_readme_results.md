# mini-infer README Results Reproduction Plan

This document maps the README claims to concrete local commands and explains what each result is meant to prove.

## 0. What This Project Proves

mini-infer is not just an HTTP wrapper around Hugging Face. Its result is:

1. Build a minimal decoder-only LLM inference engine.
2. Add production-style runtime mechanisms one by one:
   - Paged KV Cache
   - continuous batching
   - preemption and priority scheduling
   - OpenAI-compatible HTTP API
   - chunked prefill
   - prefix caching
   - True PagedAttention through `flash_attn` block tables
3. Show each mechanism with a benchmark.
4. Add independent experiments for speculative decoding, CUDA Graph, Flash Decoding, Tensor Parallelism, MLA, W8A8 quantization, PD disaggregation, and MoE Expert Parallelism.

So the README result should be read as two groups:

- Main serving path: features that are part of the actual `mini-infer-serve` route.
- Independent benchmark experiments: implemented prototypes with standalone benchmark scripts, but not necessarily enabled in the default serving path.

## 1. Local Environment

Known local model:

```bash
export MODEL_7B=/home/zsh/.cache/modelscope/hub/models/Qwen/Qwen2___5-7B-Instruct
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Current local cache check:

| Model | Local status |
| --- | --- |
| Qwen2.5-7B-Instruct | available |
| Qwen2.5-14B-Instruct | available |
| Qwen2.5-32B-Instruct | available |
| Qwen2.5-0.5B-Instruct | missing |
| Qwen2.5-1.5B-Instruct | missing |
| DeepSeek-V2-Lite | missing |

Optional GPU check before heavy runs:

```bash
nvidia-smi
```

Use a free GPU explicitly:

```bash
export CUDA_VISIBLE_DEVICES=0
```

## 2. Reproduction Waves

### Wave A: No-Weight Sanity Checks

Goal: verify the code path, scheduler, KV cache, and synthetic experiments without loading a 7B model.

```bash
PYTHONPATH=. pytest tests/test_kv_cache.py tests/test_preemption.py tests/test_engine.py tests/test_server.py -q
PYTHONPATH=. python benchmarks/benchmark_scheduler_trace.py --compare-policies
PYTHONPATH=. python benchmarks/benchmark_scheduler_trace.py --num-gpu-blocks 24 --compare-policies
PYTHONPATH=. python benchmarks/benchmark_prefix_cache.py --dry_run
PYTHONPATH=. python benchmarks/benchmark_spec.py --dry_run
```

What to look at:

- `decode_oom`: whether baseline scheduling can run out of KV blocks during decode.
- `preempt`: whether pressure-aware scheduling actually preempts lower-value requests.
- `TTFT`: time to first token.
- `ITL` / `TPOT`: inter-token latency / time per output token.
- prefix cache hit/miss behavior.

Local run on 2026-06-23:

| Check | Result |
| --- | --- |
| selected tests | `55 passed in 1.84s` |
| normal scheduler trace | all policies completed 50/50, no decode OOM |
| pressure scheduler trace, 24 KV blocks | baseline hit `decode_oom`; reserve_only, pressure_aware, adaptive completed 50/50 |
| prefix cache dry-run | pass, cache populated and hit path works |
| speculative decoding dry-run | pass, dry-run acceptance rate 100% as expected |

### Wave B: Main Serving Path With Qwen2.5-7B

Goal: reproduce the README's core path: HF baseline, mini-infer throughput, HTTP continuous batching, chunked prefill, prefix caching.

#### B1. HF Transformers Baseline

This is the reference throughput. mini-infer compares itself against this.

```bash
python benchmarks/benchmark_hf.py \
  --model "$MODEL_7B" \
  --batch-size 8 \
  --max-new-tokens 128
```

README reference: about `406 tok/s` on RTX 4090.

On A6000, exact numbers can differ. The important result is the relative comparison against mini-infer on the same machine.

Local run on 2026-06-23, physical GPU 1:

| Metric | Value |
| --- | ---: |
| Throughput | `296.8 tokens/s` |
| TTFT | `32.6 ms` |
| TPOT | `26.91 ms/token` |
| Peak memory | `15.88 GB` |

#### B2. True PagedAttention / mini-infer Throughput

This checks whether mini-infer's paged attention path can approach the HF baseline.

```bash
python benchmarks/benchmark_flash.py \
  --model "$MODEL_7B" \
  --batch-size 8 \
  --compare
```

README reference:

| Stage | Throughput |
| --- | ---: |
| serial decode | 56 tok/s |
| Paged KV Cache + batch decode | 201 tok/s |
| vectorized KV gather + DynamicCache | 361 tok/s |
| True PagedAttention | 406 tok/s |

Meaning: the project progressively closes the gap with HF by improving batching and KV access.

Local run on 2026-06-23, physical GPU 1:

| Metric | Value |
| --- | ---: |
| Throughput | `295.0 tokens/s` |
| HF-relative throughput | `99.4%` |
| TTFT | `26.1 ms` |
| TPOT | `3.39 ms/token` |
| Peak memory | `23.29 GB` |

#### B3. OpenAI-Compatible HTTP Server

Start server:

```bash
CUDA_VISIBLE_DEVICES=0 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
mini-infer-serve \
  --model "$MODEL_7B" \
  --host 0.0.0.0 \
  --port 8000 \
  --device cuda:0 \
  --dtype float16 \
  --num-gpu-blocks 256
```

Health check:

```bash
curl --noproxy '*' http://127.0.0.1:8000/healthz
```

Single request:

```bash
curl --noproxy '*' http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mini-infer","messages":[{"role":"user","content":"你好，简单介绍一下你自己"}],"stream":false,"max_tokens":128}'
```

HTTP benchmark:

```bash
python benchmarks/benchmark_server.py \
  --model "$MODEL_7B" \
  --max-tokens 64 \
  --concurrency 1 2 4 8
```

README reference: concurrent throughput from `55.7 tok/s` to `219.1 tok/s` when concurrency increases from 1 to 8.

Meaning: continuous batching combines concurrent HTTP requests into shared decode batches, so throughput rises with concurrency.

Local run on 2026-06-23, physical GPU 1:

| Concurrency | Total tokens | Time | Throughput |
| ---: | ---: | ---: | ---: |
| 1 | 65 | `1.63 s` | `39.9 tok/s` |
| 2 | 129 | `1.73 s` | `74.7 tok/s` |
| 4 | 255 | `1.90 s` | `134.2 tok/s` |
| 8 | 511 | `2.25 s` | `226.8 tok/s` |

Peak memory: `18.73 GB`.

#### B4. Chunked Prefill

```bash
python benchmarks/benchmark_chunked_prefill.py --model "$MODEL_7B" --chunk-size 256
python benchmarks/benchmark_chunked_prefill.py --model "$MODEL_7B" --chunk-size 128
```

README reference: mixed serving ITL spike drops by `57%` to `67%`.

Meaning: long prompt prefill is split into chunks, so decode requests do not wait behind one huge prefill step for too long.

Local run on 2026-06-23, physical GPU 1:

| Chunk size | No chunk ITL spike | Chunked ITL spike | Reduction | Long TTFT change | Throughput retention |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | `193.8 ms` | `74.0 ms` | `61.8%` | `193.7 -> 341.0 ms` | `103.1%` |
| 128 | `191.8 ms` | `63.4 ms` | `67.0%` | `191.8 -> 531.9 ms` | `102.0%` |

#### B5. Prefix Caching

```bash
python benchmarks/benchmark_prefix_cache.py --model "$MODEL_7B" --batch_size 8
```

README reference: shared-prefix TTFT improves by about `22%`.

Meaning: repeated system prompt / shared prefix can reuse KV blocks instead of recomputing full prefill.

Local run on 2026-06-23, physical GPU 1:

| Scenario | Miss | Hit | Result |
| --- | ---: | ---: | ---: |
| single request elapsed | `1180.3 ms` | `800.3 ms` | `1.47x` speedup |
| batch=8 elapsed | `1003.2 ms` | `1028.5 ms` | `0.98x` speedup |

Note: the single-request result reproduces the prefix-cache TTFT benefit. The batch result is noisy and uses approximate token counting, so it should not be used as the main claim.

## 3. Independent Experiments

These are real implemented experiments, but they are not all part of the default HTTP serving path.

### Speculative Decoding

Needs draft and target models. README uses Qwen2.5-0.5B draft + Qwen2.5-7B target.

```bash
python benchmarks/benchmark_spec.py --dry_run
python benchmarks/benchmark_spec.py --draft auto --target auto --K 4 --target_only
```

README reference: acceptance rate `55.85%`.

Meaning: a small draft model proposes tokens; the large target model verifies them. If accepted, decode can advance multiple tokens per target pass.

Local status on 2026-06-23: dry-run passed. Full README run is blocked because Qwen2.5-0.5B-Instruct is not cached locally.

### CUDA Graph

README uses Qwen2.5-1.5B. This local machine currently has 7B cached, but not necessarily 1.5B.

```bash
python benchmarks/benchmark_cuda_graph.py \
  --model /path/to/Qwen2.5-1.5B-Instruct \
  --batch-size 1
```

README reference: decode latency drops by `28.9%`.

Meaning: static decode steps are captured into a CUDA Graph to reduce Python / kernel launch overhead.

Local partial run on 2026-06-23, physical GPU 1, Qwen2.5-7B, batch size 1:

| Mode | Latency | Throughput |
| --- | ---: | ---: |
| eager | `24.66 ms/step` | `40.6 tok/s` |
| CUDA Graph | `23.31 ms/step` | `42.9 tok/s` |

Relative latency improvement: `5.5%`.

Note: README uses Qwen2.5-1.5B, where launch overhead is a larger fraction of total decode time. The local 7B replacement validates the path, but is not the same benchmark condition.

### Flash Decoding / Split-K Attention

This benchmark can run as a kernel-level test.

```bash
python benchmarks/benchmark_flash_decode.py
python benchmarks/benchmark_flash_decode.py --num-q-heads 28 --num-kv-heads 4
```

README reference: seq=4096 latency is `3.31x` faster than standard Triton attention, SM utilization from `9%` to `103%`.

Meaning: long-sequence single-token decode is parallelized across KV length.

Local run on 2026-06-23, physical GPU 1:

| Config | seq=4096 `flash_decode_ms` | Speed vs Phase 6.5 Triton |
| --- | ---: | ---: |
| 12 Q heads / 2 KV heads | `0.134 ms` | `2.24x` |
| 28 Q heads / 4 KV heads | `0.130 ms` | `3.28x` |

The 28Q/4KV result reproduces the README `3.31x` claim closely.

### Tensor Parallelism

README uses Qwen2.5-1.5B and 2 GPUs.

```bash
export TP_MODEL=/path/to/Qwen2.5-1.5B-Instruct
torchrun --nproc_per_node 2 benchmarks/benchmark_tp.py --model "$TP_MODEL" --mode torchrun_tp
```

README result: TP=2 greedy output exactly matches single-GPU output.

Meaning: tensor-parallel sharding is numerically correct, even if throughput is not always faster because communication overhead exists.

### MLA

```bash
python benchmarks/benchmark_mla.py --section 1
python benchmarks/benchmark_mla.py --section 2
python benchmarks/benchmark_mla.py --section 3
```

Section 1 does not need weights. Sections 2 and 3 need DeepSeek-V2-Lite.

Local run on 2026-06-23:

| Strategy | bytes/token/layer | Relative to GQA |
| --- | ---: | ---: |
| GQA | `2048` | `100.00%` |
| MLA naive | `10240` | `500.00%` |
| MLA latent | `1152` | `56.25%` |

Sections 2 and 3 are blocked locally because DeepSeek-V2-Lite is not cached.

### W8A8 Quantization

README uses Qwen2.5-1.5B.

```bash
export QUANT_MODEL=/path/to/Qwen2.5-1.5B-Instruct
python benchmarks/benchmark_quant.py --model "$QUANT_MODEL" --compare --batch-size 4
```

README reference: weight memory `3392 MB -> 2292 MB`, about `32.4%` lower; greedy token match `71.8%`.

Local status on 2026-06-23: not run under README conditions because Qwen2.5-1.5B-Instruct is not cached locally. The script can infer geometry from other models, but exact README reproduction needs the 1.5B model.

### PD Disaggregation

```bash
python benchmarks/benchmark_pd_disagg.py --section 1
python benchmarks/benchmark_pd_disagg.py --section 2
python benchmarks/benchmark_pd_disagg.py --section 3
```

README reference: TTFT split into prefill `12.3 ms`, transfer about `14.7 ms`, decode `519 ms`.

Meaning: prefill and decode can be separated into different processes/devices, then KV is transferred between them.

Local run on 2026-06-23:

| Model | seq_len | KV transfer size |
| --- | ---: | ---: |
| Qwen2.5-1.5B | 1024 | `28.00 MB` |
| Qwen2.5-7B | 1024 | `56.00 MB` |
| DeepSeek-V2-Lite MLA | 1024 | `30.38 MB` |

Sections 2 and 3 are blocked locally because the script hardcodes Qwen2.5-1.5B.

### MoE Expert Parallelism

```bash
python benchmarks/benchmark_moe.py \
  --batch-size 256 \
  --hidden-size 512 \
  --num-experts 8 \
  --top-k 2 \
  --dtype float16
```

README reference: grouped Expert Parallelism / dense throughput ratio `2.500x`.

Meaning: MoE experts can be distributed and grouped to reduce communication overhead.

Local status on 2026-06-23: blocked in this checkout. `benchmark_moe.py` imports `mini_infer.modeling.moe_layer`, `mini_infer.modeling.moe_model`, and `mini_infer.parallel.ep_engine`, but those implementation files are missing from the current working tree.

## 4. Suggested Local Execution Order

Use this order to avoid getting blocked by missing models:

1. Run Wave A first. It should finish quickly and verifies the scheduler/KV logic.
2. Run B1, B2, B3 with `$MODEL_7B`. These are the most important README results.
3. Run B4 and B5 with `$MODEL_7B`. These explain serving latency improvements.
4. Run Flash Decoding kernel benchmark; it does not depend on the 7B serving path.
5. Check whether Qwen2.5-1.5B, Qwen2.5-0.5B, and DeepSeek-V2-Lite are available. If not, download them later before CUDA Graph, Tensor Parallelism, quantization, speculative decoding, and MLA.

## 5. Result Table Template

Fill this table as each command finishes.

| Area | Script | Local status | README result | Local result | Notes |
| --- | --- | --- | --- | --- | --- |
| HF baseline | `benchmark_hf.py` | DONE | 406 tok/s | 296.8 tok/s | Qwen2.5-7B, A6000 GPU 1 |
| True PagedAttention | `benchmark_flash.py` | DONE | 406 tok/s / 100% HF | 295.0 tok/s / 99.4% HF | main throughput result, A6000 GPU 1 |
| HTTP continuous batching | `benchmark_server.py` | DONE | 55.7 -> 219.1 tok/s | 39.9 -> 226.8 tok/s | concurrency 1 -> 8, A6000 GPU 1 |
| Chunked prefill | `benchmark_chunked_prefill.py` | DONE | ITL spike -57%/-67% | -61.8% / -67.0% | chunk 256/128, A6000 GPU 1 |
| Prefix caching | `benchmark_prefix_cache.py` | DONE | TTFT -22% | single-request 1.47x speedup | shared prefix, batch result noisy |
| Speculative decoding | `benchmark_spec.py` | PARTIAL | acceptance 55.85% | dry-run passed | full run needs Qwen2.5-0.5B + 7B |
| CUDA Graph | `benchmark_cuda_graph.py` | PARTIAL | latency -28.9% | 7B bs=1 latency -5.5% | exact README condition needs Qwen2.5-1.5B |
| Flash Decoding | `benchmark_flash_decode.py` | DONE | 3.31x at seq=4096 | 3.28x at seq=4096 | kernel benchmark, A6000 GPU 1 |
| Tensor Parallelism | `benchmark_tp.py` | BLOCKED | greedy exact match | not run | exact README condition needs Qwen2.5-1.5B |
| MLA | `benchmark_mla.py` | PARTIAL | KV memory comparison | section 1 done, MLA latent = 56.25% of GQA | DeepSeek-V2-Lite needed for sections 2/3 |
| W8A8 quantization | `benchmark_quant.py` | BLOCKED | memory -32.4% | not run | exact README condition needs Qwen2.5-1.5B |
| PD disaggregation | `benchmark_pd_disagg.py` | PARTIAL | prefill/transfer/decode split | section 1 theoretical KV sizes done | sections 2/3 need Qwen2.5-1.5B |
| MoE EP | `benchmark_moe.py` | BLOCKED | EP/dense 2.500x | missing MoE implementation files | current checkout lacks `moe_layer.py` / `ep_engine.py` |

## 6. How To Read Differences From README

The README numbers were measured on Ubuntu 24.04 + 2 x RTX 4090. This machine appears to have A6000-class GPUs, so exact tok/s and latency can differ.

For reproduction, prioritize:

1. Same feature trend.
2. Same relative comparison on the same machine.
3. No functional failures.
4. Similar order of magnitude.

Exact equality is only expected for correctness tests, such as tensor-parallel greedy output matching single-GPU output.
