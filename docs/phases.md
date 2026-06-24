# 实现细节 — 核心能力模块

mini-infer 覆盖以下核心能力模块，每个模块有独立的 benchmark 脚本和可复现指标。

---

## Runtime 基础

### 最小推理链路
- 加载 Qwen2.5-7B-Instruct，串行 decode
- 建立 `LLMEngine` 和 `Request` 基础数据结构
- 与 HF baseline 对比：吞吐 56 tok/s（HF 的 13.8%）

### Paged KV Cache + Continuous Batching
- `BlockTable` + `FreeBlockPool`：block 粒度 KV 管理
- `Scheduler`：waiting / running 队列，动态准入
- Prefill / Decode 分离：单步 prefill + batch decode
- 吞吐：361 tok/s（HF 的 88.4%，优化前基准）

### Preemption + Priority Scheduling
- 高优先级请求抢占低优先级，KV blocks swap to CPU
- `swapped` 队列：swap_in / swap_out 管理
- `SamplingParams.priority` 支持请求优先级

### OpenAI Chat Completions 兼容 HTTP API
- `AsyncEngine`：后台线程 step loop + `asyncio.Queue`
- `FastAPI` server：`GET /v1/models` + `POST /v1/chat/completions`
- SSE streaming + non-streaming
- `openai_schema.py`：Pydantic 请求/响应模型

---

## 性能优化

### True PagedAttention
- `attention.py`：`PagedDecodeContext` + `patch_model_for_paged_decode`
- 将 HF attention 层替换为 `flash_attn_with_kvcache` block_table 路径
- gather_batch_kv / write_decode_kv 在 profiler 中完全消失
- **吞吐：406 tok/s（HF 的 100%）**

### Triton Decode Attention Kernel
- 从零实现 Triton decode attention kernel
- online softmax（Dao 2022）+ GQA 扩展
- 与 flash_attn 基准对比，分析 SM 利用率和 roofline
- `triton_attn.py`：独立实验性 kernel

### Chunked Prefill
- 调度器改造：`prefilling` 队列 + chunk 状态机
- 每步最多处理 `chunk_prefill_size` 个 prefill tokens
- ITL spike：**−57%**（chunk=256）/ **−67%**（chunk=128）

### Prefix Caching
- Block-level SHA-256 链式 hash
- LRU eviction + ref_count 管理
- 1-block 共享前缀：TTFT **−22%**

### Speculative Decoding
- `SpecEngine`：draft（0.5B） + target（7B）双模型
- Modified rejection sampling
- acceptance_rate：**55.85%**（K=4）

### CUDA Graph
- decode_batch 静态捕获（per batch size 一张图）
- graph pool + replay，消除 Python dispatch 开销
- 1.5B bs=1 decode 延迟 **−28.9%**

### Flash Decoding / Split-K Attention
- Triton split-K kernel：`triton_flash_decode.py`
- seq=4096：**3.31×** vs 标准 Triton kernel，SM 利用率 9%→103%
- 独立实验性 kernel，不接入主推理链路

---

## 扩展能力

### Tensor Parallelism
- NCCL all-reduce，Megatron-LM 风格权重切分
- Column/Row Parallel Linear + all-reduce hook
- TP=2，greedy 输出与单卡**完全一致**
- 注：1.5B 规模下通信开销超过计算收益，吞吐未提升；适用于单卡显存不足的大模型

### MLA（Multi-head Latent Attention）
- 三种实现：Naive / LatentCache / Absorbed
- Latent cache：KV cache 体积 **−56.25%** vs GQA
- 矩阵吸收优化（将 W_UK/W_UV 预先吸收到投影矩阵）

### Prefill/Decode 解耦（PD Disaggregation）
- 同机双进程原型，KV 序列化传输（socket）
- TTFT 三段分解：prefill 12.3ms / transfer ≈14.7ms / decode 519ms

### W8A8 量化
- `QuantLinear`：W8A8 per-channel int8
- attention 层跳过（mixed fallback）
- 权重显存 **−32.4%**（3392→2292 MB），greedy match 71.8%
- 注：decode 路径受限于 `torch._int_mm` 小 M 瓶颈，退回 FP16 fallback

### MoE + Expert Parallelism

| 实现版本 | 说明 | EP / dense |
|---------|------|-----------|
| EP padded | dense all-to-all（含 padding） | 1.954× |
| EP packed | 消除 padding，exact payload | 2.323× |
| EP grouped | 本地 expert 分组批量执行 | **2.500×** |

- `TopKRouter` + `MoELayer` + `EPMoELayer`
- True expert sharding：shard_ratio = 0.5002
- control_plane_share ≈ 1.94%，max_abs_diff = 0.000000（数值完全等价）
- runtime resident ratio：0.8334

---

完整 benchmark 数据见 [docs/benchmarks.md](benchmarks.md)。
