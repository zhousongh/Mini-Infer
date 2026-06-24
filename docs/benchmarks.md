# Benchmarks

所有 benchmark 在 Ubuntu 24.04 + 2 × RTX 4090（CUDA 12.1）上运行。
模型使用 Qwen2.5 系列（0.5B / 1.5B / 7B）及 DeepSeek-V2-Lite（MLA 实验）。

---

## 汇总：关键性能数据

| 技术 | 模型 | 指标 | 数值 |
|------|------|------|------|
| HTTP Serving（Continuous Batching） | Qwen2.5-7B | 并发 1→8 吞吐 | 55.7 → **219.1 tok/s**（3.9×） |
| True PagedAttention | Qwen2.5-7B | batch=8 吞吐 vs HF | **100.0%**（406 tok/s） |
| Chunked Prefill | Qwen2.5-7B | ITL spike 降低 | **−57%**（chunk=256）/ **−67%**（chunk=128） |
| Prefix Caching | Qwen2.5-7B | 共享前缀 TTFT | **−22%** |
| Speculative Decoding | 0.5B+7B | acceptance_rate | **55.85%** |
| CUDA Graph | Qwen2.5-1.5B | decode 延迟 bs=1 | **−28.9%** |
| Flash Decoding（split-K） | Qwen2.5-1.5B | seq=4096 延迟 vs triton | **3.31×** |
| Flash Decoding（split-K） | — | SM 利用率变化 | 9% → **103%** |
| Tensor Parallelism | Qwen2.5-1.5B | TP=2 greedy 输出 | 与单卡**完全一致** |
| MLA | DeepSeek-V2-Lite | latent cache 体积 vs GQA | **−56.25%** |
| W8A8 量化 | Qwen2.5-1.5B | 权重显存（3392→2292 MB） | **−32.4%** |
| W8A8 量化 | Qwen2.5-1.5B | greedy token match | **71.8%** |
| PD 解耦 | Qwen2.5-7B | TTFT 三段分解 | prefill 12.3ms / transfer ≈14.7ms / decode 519ms |
| MoE Expert Parallelism（grouped） | synthetic MoE | EP / dense 吞吐比 | **2.500×** |

---

## HTTP Serving 吞吐

**测试条件：** Qwen2.5-7B-Instruct，RTX 4090，max_tokens=64，非流式并发请求
**口径说明：** 使用 httpx.ASGITransport（进程内），TTFT 为近似值（等于整体响应延迟，非真实流式首 token 时间）

| 并发数 | 总 tokens | 耗时 (s) | 吞吐 (tok/s) |
|--------|-----------|----------|-------------|
| 1 | 64 | 1.15 | 55.7 |
| 2 | 128 | 1.37 | 93.1 |
| 4 | 254 | 1.46 | 174.0 |
| **8** | **510** | **2.33** | **219.1** |

并发 1→8 吞吐提升 3.9×，体现 Continuous Batching 将多个 HTTP 请求合并进同一 decode_batch 的效果。峰值显存 18.76 GB（HTTP 层不引入额外 GPU 内存）。

**复现命令：**

```bash
HF_HUB_OFFLINE=1 python benchmarks/benchmark_server.py \
  --model /path/to/Qwen2.5-7B-Instruct \
  --max-tokens 64 --trials 5 --concurrency 1 2 4 8
```

---

## 单卡吞吐演进

**测试条件：** Qwen2.5-7B-Instruct，batch=8，max_new_tokens=128，RTX 4090

| 实现 | 吞吐 | vs HF |
|------|------|-------|
| HF Transformers baseline | ~406 tok/s | 100% |
| 串行 decode（初始版） | 56 tok/s | 13.8% |
| Paged KV Cache + Batch Decode | 201 tok/s | 49.5% |
| 向量化 KV gather + DynamicCache | 361 tok/s | 88.4% |
| True PagedAttention（flash_attn block_table） | **406 tok/s** | **100.0%** |

**复现命令：**

```bash
export MODEL=/path/to/Qwen2.5-7B-Instruct && export HF_HUB_OFFLINE=1

python benchmarks/benchmark_hf.py   --model $MODEL --batch-size 8 --max-new-tokens 128
python benchmarks/benchmark_flash.py --model $MODEL --batch-size 8 --compare
```

---

## Chunked Prefill

**测试条件：** 长 prompt + 混合 decode，Qwen2.5-7B，chunk_size ∈ {128, 256}

| 配置 | ITL spike 降低 |
|------|--------------|
| chunk=256 | −57% |
| chunk=128 | −67% |

**复现命令：**

```bash
python serve.py --model $MODEL --chunk-prefill-size 256 --port 8000
python benchmarks/benchmark_chunked_prefill.py --model $MODEL --chunk-size 256
```

---

## Prefix Caching

**测试条件：** 共享前缀请求，block 级 SHA-256 hash + LRU，Qwen2.5-7B，batch=8

| 条件 | TTFT |
|------|------|
| 冷启动（cache miss） | baseline |
| 共享 1 个 block（cache hit） | **−22%** |

**复现命令：**

```bash
python benchmarks/benchmark_prefix_cache.py --dry_run           # 无需权重
python benchmarks/benchmark_prefix_cache.py --model $MODEL --batch_size 8
```

---

## Speculative Decoding

**测试条件：** 0.5B draft + 7B target，modified rejection sampling，K=4

| 指标 | 数值 |
|------|------|
| acceptance_rate | 55.85% |
| draft tokens / step | 4 |

**复现命令：**

```bash
python benchmarks/benchmark_spec.py --dry_run                               # 无需权重
python benchmarks/benchmark_spec.py --draft auto --target auto --K 4 --target_only
```

---

## CUDA Graph

**测试条件：** Qwen2.5-1.5B，batch_size=1，decode step 延迟，RTX 4090

| 模式 | decode 延迟 |
|------|------------|
| Eager | baseline |
| CUDA Graph | **−28.9%** |

**复现命令：**

```bash
python benchmarks/benchmark_cuda_graph.py \
    --model /path/to/Qwen2.5-1.5B-Instruct \
    --num-kv-heads 2 --head-dim 128 --num-layers 28
```

---

## Flash Decoding / Split-K Attention

**测试条件：** seq_len sweep，Qwen2.5-1.5B（28Q/2KV heads），RTX 4090

| seq_len | triton_65 | flash_attn | triton_splitk（split-K） |
|---------|-----------|------------|------------------------|
| 256 | ~0.04ms | ~0.03ms | ~0.07ms |
| 1024 | ~0.12ms | ~0.05ms | ~0.05ms |
| 4096 | ~0.47ms | ~0.06ms | **0.14ms**（3.31× vs triton_65） |
| 8192 | ~0.90ms | ~0.07ms | **0.26ms** |

SM 利用率（seq=4096）：triton_65 约 9% → split-K 约 103%

**复现命令：**

```bash
python benchmarks/benchmark_flash_decode.py
python benchmarks/benchmark_flash_decode.py --num-q-heads 28 --num-kv-heads 4
```

---

## Tensor Parallelism

**测试条件：** Qwen2.5-1.5B，TP=2，NCCL all-reduce，Megatron-LM 风格

| 指标 | 结果 |
|------|------|
| greedy 输出与单卡一致性 | ✅ 完全一致 |

**复现命令：**

```bash
export TP_MODEL=/path/to/Qwen2.5-1.5B-Instruct
torchrun --nproc_per_node 2 benchmarks/benchmark_tp.py --model $TP_MODEL --mode torchrun_tp
```

---

## MLA（Multi-head Latent Attention）

**测试条件：** DeepSeek-V2-Lite，MHA / GQA / MLA 三种 KV cache 方案对比

| 架构 | KV cache 大小（理论） |
|------|---------------------|
| MHA（全量 KV） | 基准 |
| GQA（4 个 KV heads） | −75%（vs MHA） |
| MLA（latent cache） | **−56.25%**（vs GQA） |

**复现命令：**

```bash
python benchmarks/benchmark_mla.py --section 1   # 理论 KV 大小（无需权重）
python benchmarks/benchmark_mla.py --section 2   # 真实显存测量（需要 DeepSeek-V2-Lite）
python benchmarks/benchmark_mla.py --section 3   # 三种实现延迟对比
```

---

## W8A8 量化

**测试条件：** Qwen2.5-1.5B，per-channel int8，attention 层跳过（mixed fallback），batch=4

| 指标 | FP16 | W8A8 | 变化 |
|------|------|------|------|
| 权重显存 | 3392 MB | 2292 MB | **−32.4%** |
| greedy token match | — | 71.8% | — |

**复现命令：**

```bash
export QUANT_MODEL=/path/to/Qwen2.5-1.5B-Instruct
python benchmarks/benchmark_quant.py --model $QUANT_MODEL --compare --batch-size 4
```

---

## PD 解耦（Disaggregated Prefill/Decode）

**测试条件：** 同机双进程，KV 序列化传输，Qwen2.5-7B

| 阶段 | 耗时 |
|------|------|
| Prefill | 12.3ms |
| KV Transfer | ≈14.7ms |
| First Decode Token | 519ms |

**复现命令：**

```bash
TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 \
    python benchmarks/benchmark_pd_disagg.py --section 3
```

---

## MoE Expert Parallelism

**测试条件：** synthetic MoE（2-GPU，2×RTX 4090），hidden_size=512，num_experts=8，top_k=2，dtype=float16

| 实现 | 吞吐（tok/s） | vs dense | 通信量 | 备注 |
|------|------------|---------|--------|------|
| Dense（1 GPU） | ~22,000 | 1.00× | — | 单卡基准 |
| EP padded（2 GPU） | ~43,000 | ~1.96× | 有 padding 冗余 | 基础 EP |
| EP packed（2 GPU） | ~52,000 | ~2.32× | 消除 padding | Non-padded dispatch |
| EP grouped（2 GPU） | **~54,700** | **2.500×** | — | Grouped execution |

其他关键指标：
- per-rank expert shard：`shard_ratio = 0.5002`（接近理想值 0.5）
- EP control plane：`control_plane_share ≈ 1.94%`
- Grouped execution：`ep_grouped_runtime_resident_ratio = 0.8334`（约 83% 的 token 在本地 rank 执行）

**复现命令：**

```bash
python benchmarks/benchmark_moe.py \
    --compare --batch-size 4 --seq-len 16 \
    --hidden-size 512 --intermediate-size 1024 \
    --num-experts 8 --top-k 2 --dtype float16 \
    --warmup 2 --runs 5 --src-rank 1
```

---

## Benchmark 口径说明

- 所有结果在 Ubuntu 24.04 + RTX 4090（CUDA 12.1）上获取，无虚拟化
- MoE Expert Parallelism benchmark 基于 **synthetic layer-level workload**，不含完整 serving 链路
- W8A8 量化的 greedy match 71.8% 表示与 FP16 基线逐 token 比较的一致率；decode 路径以 mixed fallback 为主
- PD 解耦为同机双进程原型，transfer 时间包含 socket 序列化开销
- Flash Decoding 为独立 kernel benchmark，未接入主推理链路
