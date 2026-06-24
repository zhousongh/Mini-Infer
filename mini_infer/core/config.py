"""这个文件定义引擎级基础配置，是后续真实推理实现的统一配置入口。"""

from dataclasses import dataclass


@dataclass(slots=True)
class EngineConfig:
    """保存引擎的最小配置集合。"""

    model_name: str
    device: str = "cuda:0"
    dtype: str = "float16"
    max_batch_size: int = 8
    max_model_len: int = 2048
    # flash_attn_with_kvcache 要求 block_size 必须是 256 的倍数（内核约束）
    # 256 tokens/block × 200 blocks = 51200 token 容量，适合大多数场景
    block_size: int = 256
    # KV cache 块数量（预分配 GPU tensor pool 大小）
    num_gpu_blocks: int = 200
    # 模型架构参数（必须与实际加载的模型匹配）
    # 默认值对应 Qwen2.5-7B-Instruct（GQA: 28 层，4 KV heads，128 head_dim）
    # 验证方式：cat config.json | python3 -c "import json,sys; c=json.load(sys.stdin); print(c['num_hidden_layers'], c['num_key_value_heads'], c['hidden_size']//c['num_attention_heads'])"
    num_hidden_layers: int = 28
    num_kv_heads: int = 4
    head_dim: int = 128
    tokenizer_name: str | None = None  # 若为 None，则使用 model_name
    dry_run: bool = False  # 为 True 时使用桩实现，不加载真实模型，供无 GPU 或单元测试使用
    # Phase 9：Chunked Prefill。0 = 禁用（向后兼容）；正整数 = 每步 prefill 的 token 数上限
    chunk_prefill_size: int = 0
    # Phase 12：CUDA Graph。True 时在引擎初始化后 warmup 并捕获 decode_batch 图。
    # 仅在 dry_run=False 时生效；与 chunked prefill 兼容（prefill 步走 eager 模式）。
    use_cuda_graph: bool = False
    # Phase 16：量化模式。"" = 不量化（fp16 默认路径）；"w8a8" = 第一版 W8A8
    # （activation per-row + weight per-channel，对 attention q/k/v/o_proj 保守跳过）。
    quant_mode: str = ""
    # Scheduler policy:
    # - baseline: 原始贪心准入，不做 KV 预留，同优先级不抢占
    # - reserve_only: KV 预留准入控制，只保留原始优先级抢占
    # - pressure_aware: KV 预留 + 压力感知抢占
    # - adaptive: KV 压力低时走 baseline，中压启用预留，高压启用压力感知抢占
    # - slo_aware: KV 预留 + 压力感知抢占 + priority/age/short-job waiting selection
    scheduler_policy: str = "pressure_aware"
    adaptive_reserve_threshold: float = 0.60
    adaptive_preempt_threshold: float = 0.85
    adaptive_waiting_threshold: int = 8
    scheduler_ttft_slo_steps: int = 16
    scheduler_latency_slo_steps: int = 64

    def __post_init__(self) -> None:
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size 必须大于 0")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len 必须大于 0")
        if self.block_size <= 0:
            raise ValueError("block_size 必须大于 0")
        if self.num_gpu_blocks <= 0:
            raise ValueError("num_gpu_blocks 必须大于 0")
        if self.dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("dtype 只支持 float16、bfloat16 或 float32")
        if not self.device:
            raise ValueError("device 不能为空")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers 必须大于 0")
        if self.num_kv_heads <= 0:
            raise ValueError("num_kv_heads 必须大于 0")
        if self.head_dim <= 0:
            raise ValueError("head_dim 必须大于 0")
        if self.chunk_prefill_size < 0:
            raise ValueError("chunk_prefill_size 不能小于 0")
        if self.quant_mode not in {"", "w8a8"}:
            raise ValueError(f"quant_mode 只支持 '' 或 'w8a8'，得到: {self.quant_mode!r}")
        if self.scheduler_policy not in {"baseline", "reserve_only", "pressure_aware", "adaptive", "slo_aware"}:
            raise ValueError(
                "scheduler_policy 只支持 baseline、reserve_only、pressure_aware、adaptive 或 slo_aware，"
                f"得到: {self.scheduler_policy!r}"
            )
        if not 0 <= self.adaptive_reserve_threshold <= 1:
            raise ValueError("adaptive_reserve_threshold 必须在 [0, 1] 范围内")
        if not 0 <= self.adaptive_preempt_threshold <= 1:
            raise ValueError("adaptive_preempt_threshold 必须在 [0, 1] 范围内")
        if self.adaptive_preempt_threshold < self.adaptive_reserve_threshold:
            raise ValueError("adaptive_preempt_threshold 不能小于 adaptive_reserve_threshold")
        if self.adaptive_waiting_threshold < 0:
            raise ValueError("adaptive_waiting_threshold 不能小于 0")
        if self.scheduler_ttft_slo_steps <= 0:
            raise ValueError("scheduler_ttft_slo_steps 必须大于 0")
        if self.scheduler_latency_slo_steps <= 0:
            raise ValueError("scheduler_latency_slo_steps 必须大于 0")
