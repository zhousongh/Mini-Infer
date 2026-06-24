# Roadmap

mini-infer 是一个面向 decoder-only 模型的推理引擎原型，完整实现并基准验证了从 Paged KV Cache 到 MoE Expert Parallelism 的核心推理机制。
本文件记录已明确的后续扩展方向，以及与生产级框架之间的已知 gap。

## 当前已完成的主线能力

完整能力说明见 [docs/phases.md](phases.md)。核心成果速览：

| 层次 | 代表技术 | 关键数据 |
|------|----------|---------|
| Runtime | Continuous Batching、Paged KV Cache、Chunked Prefill | batch=8 达到 HF baseline 100% |
| 性能优化 | CUDA Graph、Flash Decoding、Prefix Caching | decode 延迟 −28.9%，SM 利用率 9%→103% |
| 算法 | Speculative Decoding | acceptance rate 55.85% |
| 分布式 | Tensor Parallelism、MoE EP（Grouped Execution） | EP grouped / dense = 2.500× |
| 前沿架构 | MLA（DeepSeek-V2/V3）、PD 解耦 | latent cache −56.25% |
| 量化 | W8A8 per-channel int8 + mixed fallback | 权重显存 −32.4% |

## 下一步技术扩展方向

这些方向尚未实现，按优先级排序：

### 量化扩展

- **FP8 推理**：H100 原生 FP8 Tensor Core 支持，相比 W8A8 精度更高且硬件更匹配；当前受限于 RTX 4090 不支持 FP8。
- **Triton INT8 GEMM**：`torch._int_mm` 在 decode 路径的小 M（batch=1~8）场景下存在效率问题，导致 Phase 16 的 W8A8 decode 速度无明显提升；自写 Triton GEMM kernel 可绕过该瓶颈。
- **AWQ / GPTQ 权重加载**：复用已有量化框架的 checkpoint 格式，不需要重新量化。

### Attention 内核

- **Token-level Prefix Cache**：当前 block-level hash 粒度为 `block_size=256` tokens，命中需要完整 block 匹配；细粒度 hash 可提升短共享前缀场景的命中率。
- **Multi-block Flash Decoding**：Phase 12.5 split-K kernel 仅验证 seq=4096；更长序列（32K+）需要跨 block 并行 reduce，当前尚未实现。
- **动态 block_size**：当前 `block_size` 固定为 256（flash_attn 对齐约束），灵活 block size 需要更改 KV cache layout。

### 分布式

- **TP + EP 混合并行**：大模型实际部署中，attention 用 Tensor Parallelism（列/行切分），MoE FFN 用 Expert Parallelism（expert 分片）；当前两者独立实现，未组合使用。
- **跨机 PD 解耦**：Phase 15 是同机双进程原型，KV 传输通过共享内存；跨机需要 RDMA（InfiniBand）或高带宽以太网，涉及序列化和传输协议对齐。
- **Pipeline Parallelism 闭环**：当前 `PPEngine` 用于吞吐测量对比，未接入 continuous batching 调度器（PP + CB 组合需要 micro-batch pipeline flush 机制）。

### 服务化

- **Multi-LoRA serving**：同一基础模型下并发服务多个 LoRA adapter，需要在 KV cache 和 batch 组织上区分 adapter；适用于多任务 fine-tune 场景。
- **SLO-aware 调度**：基于 TTFT / TBT SLO 动态调整 batch size 和 preemption 策略；当前调度器只按 priority 排序，不感知延迟目标。
- **Prefix-aware 请求路由**：多副本部署下，相同系统 prompt 的请求路由到同一副本可复用 prefix cache；需要在负载均衡层增加 hash-based routing。

## 与生产级框架的已知 Gap

| 维度 | mini-infer 当前 | vLLM / TRT-LLM 生产框架 |
|------|----------------|------------------------|
| 稳定性 | 实验原型，无长期运行测试 | SLO 保障，内存泄漏监控 |
| 模型覆盖 | 仅 Qwen2.5 / DeepSeek-V2（synthetic） | 数十种架构，自动适配 |
| 量化精度 | W8A8 greedy match 71.8% | 标定工具链，PTQ/QAT |
| 多 LoRA | 未实现 | 支持 |
| 调度精细度 | 基础 priority + preemption | 完整 SLO、KV 共享感知 |
| 部署 | 单机原型 | K8s、多机 RDMA |

这些 gap 是**有意识的范围边界**：mini-infer 专注于把核心推理机制实现清楚并给出可复现的 benchmark，不追求复现完整的生产系统。

## 贡献指南

如有兴趣扩展某个方向，建议：
1. 在 [Issues](https://github.com/psmarter/mini-infer/issues) 提前说明动机和方案
2. 每个新功能对应一个 benchmark 脚本和最小测试
3. 实现前先确认与已有 Phase 数值的兼容性（greedy token match）
