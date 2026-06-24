# FAQ

---

## 环境与安装

### 如何安装 flash-attn？

`flash-attn` 需要编译，不能通过 `pyproject.toml` 的 extras 自动安装：

```bash
pip install "flash-attn>=2.5.0" --no-build-isolation
```

`--no-build-isolation` 使用系统已有的 PyTorch，避免重新编译。

### 不安装 flash-attn 能跑吗？

可以。dry-run 模式和部分 benchmark 不依赖 flash-attn。
仅 True PagedAttention（Phase 6+）路径需要 `flash_attn_with_kvcache` 的 `block_table` 参数。

### 必须用 Conda 吗？

不必须。`pip install -e ".[serve,dev]"` 在任何 Python 3.10+ 虚拟环境中都可以工作。
Makefile 默认使用系统 `python`；如需在 conda 环境中运行，使用：

```bash
PYTHON="conda run -n ai-infra python" make test-fast
```

---

## 模型与权重

### 支持哪些模型？

当前主要验证过 Qwen2.5 系列（0.5B / 1.5B / 7B）和 DeepSeek-V2-Lite（Phase 14 MLA）。
架构参数（`num_hidden_layers` / `num_kv_heads` / `head_dim`）需在 `EngineConfig` 中明确指定。

### 如何防止 transformers 联网检查？

设置环境变量：

```bash
export HF_HUB_OFFLINE=1
```

---

## 工程与架构

### block_size 为什么必须是 256 的倍数？

`flash_attn_with_kvcache` 对 KV block 对齐有要求。当前默认 `block_size=256`，不建议修改。

### Triton kernel（Phase 6.5 / 12.5）有接入主推理链路吗？

没有。两个 Triton kernel 都是**独立实验性实现**，通过各自的 benchmark 脚本验证，不替换主链路的 `flash_attn`。

Phase 6.5 的 Triton decode attention kernel 在 seq=4096 时比 `flash_attn` 慢约 5.7×，原因是 flash_attn 有高度优化的 CUDA 实现。Phase 12.5 的 Flash Decoding（split-K）专门解决长序列 decode 下 SM 利用率低（9%）的问题，达到 3.31× vs 标准 Triton 单块，但接入主链路需要更完整的 shape dispatch，当前保留为参考实现。

### Chunked Prefill 和 Prefix Caching 能同时开吗？

可以。调度器先匹配 prefix cache，命中的 block 跳过 prefill；未命中的部分按 `chunk_size` 拆分投送。两者不冲突，共同作用于降低首 token 延迟。

### Preemption 是如何触发的？

当 KV block pool 不足以容纳新请求时，调度器选择最低优先级的 running 请求，将其 KV cache swap 到 CPU，释放 GPU block 给更高优先级请求。被抢占的请求进入 swapped 队列，GPU 资源充足时重新加载（swap-in）继续 decode。

### 多请求并发时，decode batch 是怎么组的？

`LLMEngine.step()` 每次把所有 running 状态的请求合进一个 `decode_batch`，用向量化 KV gather 同时 forward。不同请求的 KV cache 存在不同的 block，通过 `block_table` 索引，不需要 padding 到同一长度。

### SpecEngine / PDEngine / TPEngine 和 LLMEngine 是什么关系？

它们是**独立的引擎扩展**，共享相同的 `ModelRunner` 和 `KVCacheManager`，但各自维护独立的生命周期。例如：
- `SpecEngine`：在 `LLMEngine` 基础上增加 draft model step
- `TPEngine`：用 `mp.spawn` + NCCL 替换单进程 forward
- `PDEngine`：拆分成两个独立的 Worker 进程

### 如何在 serving 时开启 CUDA Graph 或 W8A8 量化？

通过 CLI 标志：

```bash
mini-infer-serve --model /path/to/model --use-cuda-graph
mini-infer-serve --model /path/to/model --quant-mode w8a8
mini-infer-serve --model /path/to/model --use-cuda-graph --quant-mode w8a8
```

或通过环境变量（使用 `uvicorn` 直接启动时）：

```bash
MINI_INFER_MODEL=/path/to/model \
MINI_INFER_USE_CUDA_GRAPH=1 \
MINI_INFER_QUANT_MODE=w8a8 \
uvicorn mini_infer.serving.server:app --host 0.0.0.0 --port 8000
```

启动后可访问 `GET /healthz` 确认当前配置：

```bash
curl http://localhost:8000/healthz
# {"status":"ok","model":"...","use_cuda_graph":true,"quant_mode":"w8a8",...}
```

### MoE benchmark 为什么是 synthetic workload？

Phase 17–21 的 EP benchmark 基于 **synthetic MoE layer**（单层 forward），不包含完整 LLM serving 链路。原因：
1. 控制变量：隔离通信 vs 计算 overhead
2. 不依赖真实 MoE 模型权重（需要数十 GB）
3. 便于精确测量 ep_packed_bytes、ep_ideal_bytes 等通信口径

---

## Benchmark 与数据

### 如何复现 batch=8 吞吐达到 100% HF 的结论？

```bash
export MODEL=/path/to/Qwen2.5-7B-Instruct && export HF_HUB_OFFLINE=1
python benchmarks/benchmark_hf.py --model $MODEL --batch-size 8 --max-new-tokens 128
python benchmarks/benchmark_flash.py --model $MODEL --batch-size 8 --compare
```

### W8A8 量化的 greedy match 71.8% 是什么意思？

与 FP16 基线**逐 token 比较**的一致率。71.8% 表示约 72% 的生成 token 和 FP16 完全相同，其余有量化误差。
当前 decode 路径以 mixed fallback（int8 权重 + float32 activation）为主，精度高于纯 W8A8。

### 想看到更多数字，去哪里找？

完整 benchmark 数据与复现命令见 [docs/benchmarks.md](benchmarks.md)。

---

## 项目方向

### 这个项目会继续维护吗？

是的。当前计划是继续按阶段演进技术主线，同时逐步改善工程呈现质量。

### 可以贡献代码吗？

欢迎。建议先通过 `make test-fast` 验证环境可用，然后参考各 phase 的模块结构进行修改。
请确保改动不破坏现有测试，并为新功能添加对应的 dry_run 测试。

### 和 vLLM 的差距在哪里？

| 能力 | mini-infer | vLLM |
|------|-----------|------|
| Paged KV Cache | ✅ | ✅ |
| Continuous Batching | ✅ | ✅ |
| PagedAttention | ✅（flash_attn） | ✅（自研 CUDA kernel） |
| Chunked Prefill | ✅ | ✅ |
| Prefix Caching | ✅ | ✅ |
| Speculative Decoding | ✅ | ✅ |
| Tensor Parallelism | ✅（Megatron-LM 风格） | ✅ |
| MoE Expert Parallelism | ✅（synthetic，2-GPU） | ✅（真实模型） |
| 量化 | ✅（W8A8 原型） | ✅（AWQ/SmoothQuant/FP8） |
| 多模型 / 多租户 | ❌ | ✅ |
| production SLA | ❌ | ✅ |
| 真实 CUDA kernel | ❌（使用 flash_attn） | ✅（xFormers / FlashInfer） |
| 完整 RLHF / LoRA serving | ❌ | ✅ |

mini-infer 的核心价值不是功能覆盖度，而是**每个机制的实现路径和 benchmark 分析都清晰可追踪**。
