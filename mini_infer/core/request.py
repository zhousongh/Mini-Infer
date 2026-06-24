"""这个文件定义请求、采样参数和请求运行时状态，是调度与执行的基础数据结构。"""

from dataclasses import dataclass, field


@dataclass(slots=True)
class SamplingParams:
    """保存单个请求的最小采样参数。"""

    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens 必须大于 0")
        if self.temperature < 0:
            raise ValueError("temperature 不能小于 0")
        if not 0 < self.top_p <= 1.0:
            raise ValueError("top_p 必须在 (0, 1] 范围内")


@dataclass(slots=True)
class Request:
    """保存用户请求的静态输入信息。"""

    request_id: str
    prompt: str
    sampling_params: SamplingParams
    priority: int = 0  # 调度优先级：数值越小优先级越高（0 = 最高）


@dataclass(slots=True)
class RequestState:
    """保存请求在引擎中的运行时状态。"""

    request: Request
    prompt_token_ids: list[int] = field(default_factory=list)
    generated_token_ids: list[int] = field(default_factory=list)
    generated_text_parts: list[str] = field(default_factory=list)
    decoded_text: str = ""
    prefilled: bool = False
    finished: bool = False
    # Phase 9：Chunked Prefill 进度。记录已完成 prefill 的 token 数，0 表示未开始。
    prefilled_tokens: int = 0
    finish_reason: str | None = None
    # Phase 7：preemption 支持
    # swap_out 时将 GPU KV 拷贝到 CPU，存储于此；swap_in 时清除
    cpu_kv: list | None = None
    # swap_out 时记录的 seq_len（含 prompt + 已生成 token 数），供 swap_in 重建块分配
    swapped_seq_len: int = 0
    # Phase 10：Prefix Cache
    # 准入时命中的前缀缓存长度（按 block_size 对齐）；0 表示无命中
    prefix_cached_len: int = 0
    # 准入时命中的前缀物理 block 列表（对应 prefix_cached_len // block_size 个块）
    prefix_cached_blocks: list[int] = field(default_factory=list)
    # Scheduler observability / policy metadata.
    # arrival_step: 请求进入 waiting 队列时的 engine step，用于 SLO-aware 调度。
    arrival_step: int = 0
    # admit_step: 请求被准入 running/prefilling 时的 engine step。
    admit_step: int | None = None

    def append_generated(self, token_id: int, token_text: str) -> None:
        self.generated_token_ids.append(token_id)
        self.generated_text_parts.append(token_text)

    def mark_finished(self, reason: str) -> None:
        self.finished = True
        self.finish_reason = reason

    @property
    def output_text(self) -> str:
        if self.decoded_text:
            return self.decoded_text
        return "".join(self.generated_text_parts)
