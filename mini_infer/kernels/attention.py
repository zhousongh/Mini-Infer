"""
Phase 6 PagedAttention 模块。

通过 flash_attn_with_kvcache（2.5+）让 decode attention 直接寻址 block tensor，
消除 gather_batch_kv + write_decode_kv + DynamicCache.update 三段开销。

核心设计：
  - paged_decode_attention(): 封装 flash_attn_with_kvcache，处理 shape 转换和 GQA
  - PagedDecodeContext: 每 decode step 持有 block_table / cache_seqlens 的共享可变容器
  - patch_model_for_paged_decode(): 永久 patch Qwen2 各 attention 层：
      - ctx.block_table is None → prefill 路径，回退到原始 HF forward
      - ctx.block_table 已设置 → decode 路径，绕过 DynamicCache，直接用 flash_attn_with_kvcache

RoPE 策略：
  - 在 patch 内手动调用 attn_module.rotary_emb(v, seq_len=max_kv_len) + apply_rotary_pos_emb
  - 不使用 flash_attn 内置 rotary_cos/sin 参数（避免与 Qwen2 格式不兼容）
  - max_kv_len = max(cache_seqlens) + 1，确保 cos/sin 覆盖当前最大 token 位置

注意：
  - 本模块仅适用于 Qwen2/Qwen2.5 系列模型（依赖 Qwen2Attention 内部结构）
  - flash_attn_with_kvcache 调用后，k_cache/v_cache 已 in-place 写入新 KV，
    调用方无需再调用 write_decode_kv（只需 advance_seq_lens）
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

try:
    from flash_attn import flash_attn_with_kvcache
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False

if TYPE_CHECKING:
    from .kv_cache import KVCacheManager


class PagedDecodeContext:
    """
    每个 decode step 的共享状态容器，所有 patch 后的 attention 层都读取此对象。

    使用流程：
        1. decode_batch() 调用 ctx.set(block_table, cache_seqlens, max_kv_len)
        2. model.forward() 执行，patched attention 读取 ctx 中的张量（无额外 .item() sync）
        3. decode_batch() 调用 ctx.clear()

    ctx.block_table is None 时，patched forward 回退到原始 HF forward（prefill 路径）。
    """

    def __init__(self) -> None:
        self.block_table: torch.Tensor | None = None
        self.cache_seqlens: torch.Tensor | None = None
        # max_kv_len 由 decode_batch() 预先计算（一次 CPU-GPU sync），
        # 避免在 28 层 patched_forward 里各调用一次 .item()（原因：每次 .item() 触发一次同步）
        self.max_kv_len: int = 0

    def set(
        self,
        block_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_kv_len: int,
    ) -> None:
        self.block_table = block_table
        self.cache_seqlens = cache_seqlens
        self.max_kv_len = max_kv_len

    def clear(self) -> None:
        self.block_table = None
        self.cache_seqlens = None
        self.max_kv_len = 0


def paged_decode_attention(
    q: torch.Tensor,
    k_new: torch.Tensor,
    v_new: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
) -> torch.Tensor:
    """
    封装 flash_attn_with_kvcache，执行 paged decode attention。

    Args:
        q:             [batch, 1, num_q_heads, head_dim]  — 已经过 RoPE
        k_new:         [batch, 1, num_kv_heads, head_dim] — 已经过 RoPE
        v_new:         [batch, 1, num_kv_heads, head_dim]
        k_cache:       [num_blocks, block_size, num_kv_heads, head_dim]（in-place 更新）
        v_cache:       同上（in-place 更新）
        block_table:   [batch, max_blocks] int32
        cache_seqlens: [batch] int32，新 token 写入前的 cache 长度

    Returns:
        attn_out: [batch, 1, num_q_heads, head_dim]

    副作用：k_new/v_new 被 flash_attn in-place 写入 k_cache/v_cache 的对应位置。
    """
    if not _FLASH_ATTN_AVAILABLE:
        raise ImportError(
            "flash_attn is required for PagedAttention. "
            "Install with: pip install 'flash-attn>=2.5.0' --no-build-isolation"
        )
    head_dim = q.shape[-1]
    softmax_scale = 1.0 / math.sqrt(head_dim)
    return flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k_new,
        v=v_new,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
        softmax_scale=softmax_scale,
    )


def patch_model_for_paged_decode(
    model: torch.nn.Module,
    kv_manager: KVCacheManager,
) -> PagedDecodeContext:
    """
    永久 patch Qwen2 模型的所有 attention 层，decode 时使用 paged attention。

    实现方式：
      - 替换每层 self_attn.forward 为 patched_forward
      - patched_forward 检查 ctx.block_table：
          None   → 调用原始 HF forward（prefill 路径，不影响 KV cache 写入逻辑）
          非 None → paged decode 路径（绕过 DynamicCache，直接用 flash_attn_with_kvcache）

    RoPE 细节：
      - rotary_emb(v, seq_len=max_kv_len) 返回 cos/sin: [max_kv_len, head_dim]
      - apply_rotary_pos_emb(q, k, cos, sin, position_ids) 用 position_ids=[batch,1] 索引
      - max_kv_len = max(cache_seqlens) + 1，确保覆盖当前最大位置

    Returns:
        PagedDecodeContext: 调用方在每次 decode forward 前 set()，forward 后 clear()。
    """
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    ctx = PagedDecodeContext()

    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        k_cache = kv_manager.k_cache[layer_idx]
        v_cache = kv_manager.v_cache[layer_idx]
        original_forward = attn.forward

        def _make_patched_forward(
            attn_module: torch.nn.Module,
            k_c: torch.Tensor,
            v_c: torch.Tensor,
            orig_fwd,
        ):
            def patched_forward(
                hidden_states: torch.Tensor,
                attention_mask=None,
                position_ids=None,
                past_key_value=None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position=None,
                **kwargs,
            ):
                # prefill 路径：ctx.block_table 未设置，使用原始 HF forward
                if ctx.block_table is None:
                    return orig_fwd(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        **kwargs,
                    )

                # decode 路径：paged attention
                bsz, q_len, _ = hidden_states.shape

                # 1. q/k/v 投影
                q = attn_module.q_proj(hidden_states)
                k = attn_module.k_proj(hidden_states)
                v = attn_module.v_proj(hidden_states)

                # 2. reshape → [batch, num_heads, seq=1, head_dim]（HF 约定）
                q = q.view(bsz, q_len, attn_module.num_heads, attn_module.head_dim).transpose(1, 2)
                k = k.view(bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim).transpose(1, 2)
                v = v.view(bsz, q_len, attn_module.num_key_value_heads, attn_module.head_dim).transpose(1, 2)

                # 3. RoPE：cos/sin 需要覆盖到当前批次的最大 token 位置
                #    max_kv_len 由 decode_batch() 预计算并存入 ctx（避免每层调用 .item() 同步）
                cos, sin = attn_module.rotary_emb(v, seq_len=ctx.max_kv_len)
                q, k = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

                # 4. 转换为 flash_attn 格式：[batch, seq=1, num_heads, head_dim]
                q_fa = q.transpose(1, 2)
                k_fa = k.transpose(1, 2)
                v_fa = v.transpose(1, 2)

                # 5. Paged decode attention（同时 in-place 写入 k_fa/v_fa 到 block cache）
                attn_out = paged_decode_attention(
                    q_fa, k_fa, v_fa,
                    k_c, v_c,
                    ctx.block_table, ctx.cache_seqlens,
                )

                # 6. 输出投影：[batch, 1, num_heads * head_dim] → [batch, 1, hidden_size]
                attn_out = attn_out.reshape(bsz, q_len, -1)
                attn_out = attn_module.o_proj(attn_out)

                return attn_out, None, None

            return patched_forward

        attn.forward = _make_patched_forward(attn, k_cache, v_cache, original_forward)

    return ctx
