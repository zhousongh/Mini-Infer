# Architecture

mini-infer 的整体架构围绕 **LLMEngine** 展开，逐步扩展出分布式、量化、MoE EP 等能力。

## 包结构

```
mini_infer/
├── core/           # 基础数据结构：EngineConfig、Request、SamplingParams
├── cache/          # Paged KV Cache：BlockTable、FreeBlockPool、Prefix Cache、KV 传输
├── kernels/        # Attention kernel：PagedAttention、Triton decode、Flash Decoding
├── modeling/       # 模型执行：ModelRunner、量化、MLA、MoE
├── runtime/        # 推理引擎：LLMEngine、AsyncEngine、Scheduler、SpecEngine、PDEngine
├── parallel/       # 分布式：TP、EP、Replica、PP
├── serving/        # HTTP 服务：OpenAI 兼容接口、FastAPI app
└── cli/            # CLI 入口：mini-infer-serve、mini-infer-chat、mini-infer-demo
```

## 请求生命周期

```
HTTP 请求 / Python API
        │
        ▼
  AsyncEngine              serving/  — 后台线程 step loop + asyncio.Queue
        │
        ▼
  LLMEngine                runtime/  — continuous batching 主循环
    ├── Scheduler           runtime/  — waiting/running/swapped/prefilling 队列
    │                                   Preemption、Priority、Chunked Prefill
    ├── KVCacheManager      cache/    — BlockTable + FreeBlockPool
    │                                   block-level SHA-256 hash + LRU Prefix Cache
    └── ModelRunner         modeling/ — prefill / decode_batch
            ├── PagedAttention  kernels/  — flash_attn_with_kvcache + block_table
            ├── CUDA Graph      modeling/ — decode step 静态捕获
            └── QuantLinear     modeling/ — W8A8 per-channel int8 / mixed fallback
```

## 模块说明

### `mini_infer.core` — 基础数据结构

| 模块 | 职责 |
|------|------|
| `core.config` | `EngineConfig` dataclass：block_size / max_blocks / num_kv_heads / head_dim 等全部引擎参数 |
| `core.request` | `Request` / `RequestState` / `SamplingParams`：请求数据结构和完整生命周期状态 |

### `mini_infer.cache` — KV Cache 管理

| 模块 | 职责 |
|------|------|
| `cache.kv_cache` | `KVCacheManager`：BlockTable + FreeBlockPool；Phase 10 集成 Prefix Cache（block-level SHA-256 链式 hash + LRU + ref_count） |
| `cache.kv_transfer` | KV 序列化传输辅助（Phase 15 PD 解耦使用） |

### `mini_infer.kernels` — Attention Kernels

| 模块 | 职责 |
|------|------|
| `kernels.attention` | `PagedDecodeContext` + `patch_model_for_paged_decode`：将 HF 模型 attention 替换为 `flash_attn_with_kvcache` block_table 路径 |
| `kernels.triton_attn` | Triton decode attention kernel（Phase 6.5）：online softmax + GQA，对应 Phase 6.5 benchmark |
| `kernels.triton_flash_decode` | Flash Decoding split-K kernel（Phase 12.5）：seq=4096 时 3.31× vs 标准 Triton，SM 利用率 9%→103% |

### `mini_infer.modeling` — 模型执行层

| 模块 | 职责 |
|------|------|
| `modeling.model_runner` | `ModelRunner`：prefill / decode_batch 执行路径；Phase 12 集成 CUDA Graph 静态捕获 |
| `modeling.quantization` | `QuantLinear`（W8A8 per-channel int8）、`quantize_model`、mixed fallback；1.5B 权重显存 −32.4% |
| `modeling.mla_attention` | MLA 三种实现：`Naive` / `LatentCache` / `Absorbed`；DeepSeek-V2/V3 架构，latent cache 体积 −56.25% vs GQA |
| `modeling.moe_layer` | `TopKRouter` / `MoELayer` / `EPMoELayer`；Phase 17 基础 EP → Phase 21 grouped local execution |
| `modeling.moe_model` | `SyntheticMoEConfig` / `SyntheticMoEModel`：benchmark 专用 synthetic MoE |

### `mini_infer.runtime` — 推理引擎

| 模块 | 职责 |
|------|------|
| `runtime.engine` | `LLMEngine`：continuous batching 主循环；`step()` 每轮驱动一次 prefill + decode |
| `runtime.scheduler` | `Scheduler`：waiting / running / swapped / prefilling 四队列；preemption swap（Phase 7）；chunked prefill（Phase 9） |
| `runtime.async_engine` | `AsyncEngine`：后台线程 step loop + `asyncio.Queue`；Phase 8 HTTP serving 的异步前端 |
| `runtime.spec_engine` | `SpecEngine`：draft + target 双模型 Speculative Decoding（Phase 11）；modified rejection sampling，acceptance_rate 55.85% |
| `runtime.pd_engine` | `PDEngine`：Disaggregated Prefill/Decode 入口（Phase 15）；同机双进程原型 |
| `runtime.pd_worker` | `PrefillWorker` / `DecodeWorker` + KV 传输；TTFT 三段分解：prefill 12.3ms / transfer ≈14.7ms / decode 519ms |

### `mini_infer.parallel` — 分布式扩展

| 模块 | 职责 |
|------|------|
| `parallel.tp_engine` | `TPEngine`：真 TP（NCCL all-reduce，Phase 13）；`mp.spawn` + 文件锁 rendezvous |
| `parallel.tp_model_runner` | `TensorParallelModelRunner`：Megatron-LM 风格 column/row parallel + all-reduce hook |
| `parallel.ep_engine` | `EPEngine`：2-GPU Expert Parallelism（Phase 17–21）；padded → packed → grouped 三阶段优化，最终 2.500× vs dense |
| `parallel.replica_engine` | `ReplicaEngine`：数据并行副本（Phase 4） |
| `parallel.pp_engine` | `PPEngine`：HF Pipeline Parallel（Phase 4，吞吐测量用） |

### `mini_infer.serving` — HTTP 服务层

| 模块 | 职责 |
|------|------|
| `serving.server` | FastAPI HTTP server；`GET /v1/models`，`POST /v1/chat/completions`（streaming + non-streaming） |
| `serving.openai_schema` | `ChatCompletionRequest` / `ChatCompletionResponse` Pydantic 模型 |
| `clients.chat_client` | 交互式 CLI 聊天客户端，支持 dry-run 自动启动临时服务 |

### `mini_infer.cli` — CLI 入口

| 命令 | 等价脚本 | 职责 |
|------|----------|------|
| `mini-infer-serve` | `python serve.py` | 启动 OpenAI 兼容 HTTP 服务 |
| `mini-infer-chat` | `python quick_chat.py` | 交互式聊天（干运行 / 真实模型） |
| `mini-infer-demo` | `python demo.py` | 功能对比演示（量化 / CUDA Graph / Prefix Cache） |

## 向后兼容

重构前的扁平路径（如 `from mini_infer.engine import LLMEngine`）通过同名 shim 文件保持可用，不需要修改任何现有代码。新代码推荐使用规范子包路径：

```python
# 旧路径（仍可用）
from mini_infer.engine import LLMEngine
from mini_infer.config import EngineConfig

# 新路径（推荐）
from mini_infer.runtime.engine import LLMEngine
from mini_infer.core.config import EngineConfig
```

## 关键设计选择

### BlockTable vs 连续 KV Buffer

Phase 1–2 使用连续 KV buffer，Phase 6 切换到 `flash_attn_with_kvcache` 的 `block_table` 路径。block table 允许不同长度请求的 KV 块共享物理内存、Prefix Cache 的 block-level 复用、以及 Speculative Decoding 的高效 KV 管理。

### Continuous Batching 状态机

`Scheduler.step()` 每轮返回一批 `running` 请求，其中部分处于 prefill 阶段，其余处于 decode 阶段。`ModelRunner` 根据请求状态分别调用 `prefill()` 和 `decode_batch()`，实现请求粒度的动态批处理。

### CUDA Graph 的适用范围

CUDA Graph 仅对 decode_batch 有效（输入 shape 固定），不适用于 prefill（序列长度动态变化）。Phase 12 的实现对每种 batch size 分别捕获静态图，1.5B bs=1 延迟 −28.9%。

### MoE EP 通信演进

- Phase 17：dense all-to-all（padded，有冗余通信）
- Phase 19：packed dispatch（ep_packed_bytes = ep_ideal_bytes，消除 padding）
- Phase 21：grouped local execution（**2.500× vs dense**）

## 后续扩展方向

详见 [docs/roadmap.md](roadmap.md)。
