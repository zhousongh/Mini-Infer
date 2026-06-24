"""
Phase 15：KV Cache 传输层。

提供 Prefill Worker → Decode Worker 的 KV block 数据传输：
- KVPayload：序列化后的 KV 数据（tensor bytes + 元数据）
- KVSender / KVReceiver：基于 multiprocessing.Queue 的传输接口
  （同机进程间，零拷贝语义；Queue 内部用 pickle，适合同机共享内存场景）

设计约束：
- 只传 KV tensor 数据，不传 KVCacheManager 状态（block_table 由 decode 侧重新分配）
- 传输单元是"一个请求的所有层 KV"，格式：list[tuple[Tensor, Tensor]]（每层 k, v）
- dry_run 模式下 KV tensor 为 None，传输层仍可正常工作（传 None）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing import Queue
from typing import Optional

import torch


@dataclass
class KVPayload:
    """单个请求的 KV 传输载荷。"""

    request_id: str
    # 每层 (k_tensor, v_tensor)；dry_run 时为空列表
    # k shape: [seq_len, num_kv_heads, head_dim]
    # v shape: [seq_len, num_kv_heads, head_dim]
    kv_layers: list[tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = field(default_factory=list)
    # prefill 阶段生成的第一个 token id
    first_token_id: int = 0
    # prompt token 数（decode 侧重建 seq_len 用）
    prompt_len: int = 0
    # 采样参数透传
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    # 计时：prefill 耗时（秒）
    prefill_time: float = 0.0


class KVSender:
    """向 decode worker 发送 KV payload。"""

    def __init__(self, queue: Queue) -> None:
        self._q = queue

    def send(self, payload: KVPayload) -> None:
        self._q.put(payload)


class KVReceiver:
    """从 prefill worker 接收 KV payload。"""

    def __init__(self, queue: Queue) -> None:
        self._q = queue

    def recv(self, timeout: float = 60.0) -> KVPayload:
        payload = self._q.get(timeout=timeout)
        return payload


def extract_kv_from_past(
    past_key_values,  # HF DynamicCache or list[tuple[Tensor, Tensor]]
    seq_len: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    从 HF past_key_values 提取每层 KV tensor，裁剪到 seq_len。

    返回：list of (k, v)，每个 shape [seq_len, num_kv_heads, head_dim]
    （去掉 batch 维度，batch=1）
    """
    result = []
    # DynamicCache: .key_cache[layer] shape [batch, num_kv_heads, seq_len, head_dim]
    # tuple list: [(k, v), ...] 同上
    try:
        num_layers = len(past_key_values.key_cache)
        for i in range(num_layers):
            k = past_key_values.key_cache[i][0, :, :seq_len, :].permute(1, 0, 2).cpu()  # [seq, heads, dim]
            v = past_key_values.value_cache[i][0, :, :seq_len, :].permute(1, 0, 2).cpu()
            result.append((k, v))
    except AttributeError:
        # fallback: list of (k, v) tuples
        for k_layer, v_layer in past_key_values:
            k = k_layer[0, :, :seq_len, :].permute(1, 0, 2).cpu()
            v = v_layer[0, :, :seq_len, :].permute(1, 0, 2).cpu()
            result.append((k, v))
    return result


def measure_kv_size_bytes(kv_layers: list[tuple[torch.Tensor, torch.Tensor]]) -> int:
    """计算 KV payload 的总字节数（用于 benchmark 报告）。"""
    total = 0
    for k, v in kv_layers:
        if k is not None:
            total += k.nelement() * k.element_size()
        if v is not None:
            total += v.nelement() * v.element_size()
    return total
