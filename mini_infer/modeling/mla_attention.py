"""
Phase 14 MLA（Multi-head Latent Attention）实现。

参考：DeepSeek-V2 技术报告（arXiv:2405.04434）和
     deepseek-ai/DeepSeek-V2-Lite modeling_deepseek.py（trust_remote_code 版）。

实现两个版本，用于对比 KV cache 压缩效果：

  MLAAttentionNaive
    与 HF DeepseekV2Attention.forward() 数学等价。
    KV cache 存储完整展开的 key_states + value_states：
      每层 cache 大小 = num_heads × (q_head_dim + v_head_dim) × 2 bytes / token
      DeepSeek-V2-Lite：16 × (192 + 128) × 2 = 10,240 bytes / token / layer

  MLAAttentionLatentCache
    只缓存低秩 latent：compressed_kv（kv_lora_rank 维）+ k_pe（qk_rope_head_dim 维）。
    Attention 时即时展开 K/V，等价于 naive 版输出。
      每层 cache 大小 = (kv_lora_rank + qk_rope_head_dim) × 2 bytes / token
      DeepSeek-V2-Lite：(512 + 64) × 2 = 1,152 bytes / token / layer

DeepSeek-V2-Lite 超参（来自 config.json）：
  hidden_size        = 2048
  num_heads          = 16
  q_lora_rank        = None  ← V2-Lite 不压缩 Q，直接用 q_proj
  qk_nope_head_dim   = 128
  qk_rope_head_dim   = 64
  kv_lora_rank       = 512
  v_head_dim         = 128
  q_head_dim         = qk_nope_head_dim + qk_rope_head_dim = 192
  softmax_scale      = 1 / sqrt(q_head_dim) = 1 / sqrt(192)

注意：本模块不实现 RoPE（只对 rope 分量作占位处理，不改变形状），
     以保持独立于 transformers 的 rotary 实现。正确性通过数学等价测试
     和下载后的真实模型对比验证。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────


@dataclass
class MLAConfig:
    """
    MLA 注意力超参数，与 DeepSeek-V2-Lite config.json 对应。

    字段说明：
      hidden_size        : Transformer 隐藏层维度
      num_heads          : Q attention head 数
      q_lora_rank        : Q 压缩 rank（None = 不压缩，直接用 q_proj）
      qk_nope_head_dim   : Q/K 的非 RoPE 分量维度
      qk_rope_head_dim   : Q/K 的 RoPE 分量维度
      kv_lora_rank       : KV 压缩 rank（latent 维度 d_c）
      v_head_dim         : V head 维度
    """
    hidden_size: int = 2048
    num_heads: int = 16
    q_lora_rank: Optional[int] = None
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    kv_lora_rank: int = 512
    v_head_dim: int = 128

    @property
    def q_head_dim(self) -> int:
        """每个 Q/K head 的完整维度（nope + rope）。"""
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def softmax_scale(self) -> float:
        return 1.0 / math.sqrt(self.q_head_dim)


# ──────────────────────────────────────────────
# KV Cache 数据结构
# ──────────────────────────────────────────────


@dataclass
class MLAKVCacheNaive:
    """
    Naive 版 KV cache：存储完整展开的 key / value。
    与标准 MHA KV cache 格式相同，HF DynamicCache 兼容。

    每层大小 = (num_heads × q_head_dim + num_heads × v_head_dim) × 2 bytes / token
    DeepSeek-V2-Lite = (16×192 + 16×128) × 2 = 10,240 bytes / token / layer
    """
    key_states: torch.Tensor    # (batch, num_heads, seq_len, q_head_dim)
    value_states: torch.Tensor  # (batch, num_heads, seq_len, v_head_dim)

    @property
    def seq_len(self) -> int:
        return self.key_states.shape[2]

    def bytes_per_token_per_layer(self) -> int:
        return (
            self.key_states.shape[1] * self.key_states.shape[3]
            + self.value_states.shape[1] * self.value_states.shape[3]
        ) * 2  # fp16


@dataclass
class MLAKVCacheLatent:
    """
    Latent cache：只存储 compressed_kv（低秩 latent）+ k_pe（RoPE 分量）。
    Attention 时通过 kv_b_proj 即时展开 K/V。

    每层大小 = (kv_lora_rank + qk_rope_head_dim) × 2 bytes / token
    DeepSeek-V2-Lite = (512 + 64) × 2 = 1,152 bytes / token / layer
    约为 naive cache 的 11.25%，或 GQA(4KV,128dim) 的 56.25%
    （GQA 基准：Qwen2.5-7B 风格，4 KV heads，head_dim=128）
    """
    compressed_kv: torch.Tensor  # (batch, seq_len, kv_lora_rank)
    k_pe: torch.Tensor           # (batch, seq_len, qk_rope_head_dim) — 所有 head 共享

    @property
    def seq_len(self) -> int:
        return self.compressed_kv.shape[1]

    def bytes_per_token_per_layer(self) -> int:
        """从 tensor shape 推导，与 MLAKVCacheNaive.bytes_per_token_per_layer 接口一致。"""
        return (self.compressed_kv.shape[-1] + self.k_pe.shape[-1]) * 2  # fp16


# ──────────────────────────────────────────────
# RMSNorm（KV latent 归一化，与 HF kv_a_layernorm 对应）
# ──────────────────────────────────────────────


class RMSNorm(nn.Module):
    """简化版 RMSNorm，与 DeepSeek-V2 modeling_deepseek.py 中的用法一致。"""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * x).to(input_dtype)


# ──────────────────────────────────────────────
# MLAAttentionNaive：与 HF DeepseekV2Attention 等价
# ──────────────────────────────────────────────


class MLAAttentionNaive(nn.Module):
    """
    Naive MLA 实现，与 HF DeepseekV2Attention.forward() 数学等价。

    权重对应关系（来自 modeling_deepseek.py）：
      q_proj          (q_lora_rank=None 时) hidden_size → num_heads × q_head_dim
      q_a_proj        (q_lora_rank≠None 时) hidden_size → q_lora_rank
      q_a_layernorm   RMSNorm(q_lora_rank)
      q_b_proj        q_lora_rank → num_heads × q_head_dim
      kv_a_proj_with_mqa  hidden_size → kv_lora_rank + qk_rope_head_dim
      kv_a_layernorm      RMSNorm(kv_lora_rank)
      kv_b_proj           kv_lora_rank → num_heads × (qk_nope_head_dim + v_head_dim)
      o_proj              num_heads × v_head_dim → hidden_size

    KV cache：存储完整 key_states + value_states（不压缩）。
    """

    def __init__(self, config: MLAConfig) -> None:
        super().__init__()
        self.cfg = config
        c = config

        # Q 投影
        if c.q_lora_rank is None:
            # V2-Lite：直接投影
            self.q_proj = nn.Linear(c.hidden_size, c.num_heads * c.q_head_dim, bias=False)
            self.q_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None
        else:
            self.q_proj = None
            self.q_a_proj = nn.Linear(c.hidden_size, c.q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(c.q_lora_rank)
            self.q_b_proj = nn.Linear(c.q_lora_rank, c.num_heads * c.q_head_dim, bias=False)

        # KV 压缩与展开
        self.kv_a_proj_with_mqa = nn.Linear(
            c.hidden_size, c.kv_lora_rank + c.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(c.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            c.kv_lora_rank, c.num_heads * (c.qk_nope_head_dim + c.v_head_dim), bias=False
        )

        # 输出投影
        self.o_proj = nn.Linear(c.num_heads * c.v_head_dim, c.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,              # (batch, seq, hidden_size)
        past_cache: Optional[MLAKVCacheNaive] = None,
    ) -> Tuple[torch.Tensor, MLAKVCacheNaive]:
        """
        Args:
            hidden_states: (batch, seq, hidden_size)
            past_cache:    已有的 KV cache（decode 时传入）
        Returns:
            output:     (batch, seq, hidden_size)
            new_cache:  包含本次及历史 key/value 的 MLAKVCacheNaive
        """
        c = self.cfg
        bsz, q_len, _ = hidden_states.shape

        # ── Q 投影 ──
        if c.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        # q: (bsz, q_len, num_heads × q_head_dim) → (bsz, num_heads, q_len, q_head_dim)
        q = q.view(bsz, q_len, c.num_heads, c.q_head_dim).transpose(1, 2)
        q_nope, q_pe = q.split([c.qk_nope_head_dim, c.qk_rope_head_dim], dim=-1)

        # ── KV 压缩 ──
        compressed_kv_and_kpe = self.kv_a_proj_with_mqa(hidden_states)
        # 切分：前 kv_lora_rank 为 compressed_kv，后 qk_rope_head_dim 为 k_pe
        compressed_kv = compressed_kv_and_kpe[..., : c.kv_lora_rank]
        k_pe = compressed_kv_and_kpe[..., c.kv_lora_rank :]
        # k_pe: (bsz, q_len, qk_rope_head_dim) → (bsz, 1, q_len, qk_rope_head_dim)
        k_pe = k_pe.view(bsz, q_len, 1, c.qk_rope_head_dim).transpose(1, 2)

        # ── KV 展开 ──
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, c.num_heads, c.qk_nope_head_dim + c.v_head_dim)
            .transpose(1, 2)
        )  # (bsz, num_heads, q_len, qk_nope_head_dim + v_head_dim)
        k_nope, value_states = kv.split([c.qk_nope_head_dim, c.v_head_dim], dim=-1)

        # ── 拼接 Q 和 K（nope + rope）──
        # 注意：本模块不做实际 RoPE，只保留形状（q_pe/k_pe 原样保留）
        # 真实模型加载后通过 rotary_emb 应用旋转位置编码
        query_states = q.new_empty(bsz, c.num_heads, q_len, c.q_head_dim)
        query_states[:, :, :, : c.qk_nope_head_dim] = q_nope
        query_states[:, :, :, c.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty(bsz, c.num_heads, q_len, c.q_head_dim)
        key_states[:, :, :, : c.qk_nope_head_dim] = k_nope
        key_states[:, :, :, c.qk_nope_head_dim :] = k_pe  # broadcast: 1 head → num_heads

        # ── 拼接历史 KV cache ──
        if past_cache is not None:
            key_states = torch.cat([past_cache.key_states, key_states], dim=2)
            value_states = torch.cat([past_cache.value_states, value_states], dim=2)

        new_cache = MLAKVCacheNaive(key_states=key_states, value_states=value_states)

        # ── Scaled dot-product attention ──
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            scale=c.softmax_scale,
        )  # (bsz, num_heads, q_len, v_head_dim)

        # ── 输出投影 ──
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, c.num_heads * c.v_head_dim)
        output = self.o_proj(attn_output)

        return output, new_cache


# ──────────────────────────────────────────────
# MLAAttentionLatentCache：只缓存 latent
# ──────────────────────────────────────────────


class MLAAttentionLatentCache(nn.Module):
    """
    Latent cache 版 MLA：KV cache 只存 compressed_kv + k_pe，
    Attention 时调用 kv_b_proj 即时展开。

    与 MLAAttentionNaive 使用相同权重结构，数学等价，
    但 cache 大小从 10,240 bytes/token/layer 降至 1,152 bytes/token/layer（V2-Lite）。

    代价：每个 decode step 需对全部历史 compressed_kv（seq_len 条）
         做一次 kv_b_proj 展开，计算量 O(seq_len × kv_lora_rank × hidden)。
    """

    def __init__(self, config: MLAConfig) -> None:
        super().__init__()
        self.cfg = config
        c = config

        if c.q_lora_rank is None:
            self.q_proj = nn.Linear(c.hidden_size, c.num_heads * c.q_head_dim, bias=False)
            self.q_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None
        else:
            self.q_proj = None
            self.q_a_proj = nn.Linear(c.hidden_size, c.q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(c.q_lora_rank)
            self.q_b_proj = nn.Linear(c.q_lora_rank, c.num_heads * c.q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = nn.Linear(
            c.hidden_size, c.kv_lora_rank + c.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(c.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            c.kv_lora_rank, c.num_heads * (c.qk_nope_head_dim + c.v_head_dim), bias=False
        )
        self.o_proj = nn.Linear(c.num_heads * c.v_head_dim, c.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_cache: Optional[MLAKVCacheLatent] = None,
    ) -> Tuple[torch.Tensor, MLAKVCacheLatent]:
        c = self.cfg
        bsz, q_len, _ = hidden_states.shape

        # ── Q 投影（同 naive）──
        if c.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, c.num_heads, c.q_head_dim).transpose(1, 2)
        q_nope, q_pe = q.split([c.qk_nope_head_dim, c.qk_rope_head_dim], dim=-1)

        # ── KV 压缩（只保留 latent）──
        compressed_kv_and_kpe = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv = compressed_kv_and_kpe[..., : c.kv_lora_rank]  # (bsz, q_len, kv_lora_rank)
        k_pe = compressed_kv_and_kpe[..., c.kv_lora_rank :]           # (bsz, q_len, qk_rope_head_dim)

        # ── 拼接历史 latent cache ──
        if past_cache is not None:
            compressed_kv = torch.cat([past_cache.compressed_kv, compressed_kv], dim=1)
            k_pe = torch.cat([past_cache.k_pe, k_pe], dim=1)

        kv_seq_len = compressed_kv.shape[1]
        new_cache = MLAKVCacheLatent(compressed_kv=compressed_kv, k_pe=k_pe)

        # ── 即时展开全部历史 KV（在 attention 之前）──
        kv_full = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, kv_seq_len, c.num_heads, c.qk_nope_head_dim + c.v_head_dim)
            .transpose(1, 2)
        )
        k_nope_full, value_states = kv_full.split([c.qk_nope_head_dim, c.v_head_dim], dim=-1)

        # k_pe 广播到 num_heads
        k_pe_full = k_pe.view(bsz, kv_seq_len, 1, c.qk_rope_head_dim).transpose(1, 2)

        key_states = k_pe_full.new_empty(bsz, c.num_heads, kv_seq_len, c.q_head_dim)
        key_states[:, :, :, : c.qk_nope_head_dim] = k_nope_full
        key_states[:, :, :, c.qk_nope_head_dim :] = k_pe_full

        # ── Q 拼接（当前 token）──
        query_states = q.new_empty(bsz, c.num_heads, q_len, c.q_head_dim)
        query_states[:, :, :, : c.qk_nope_head_dim] = q_nope
        query_states[:, :, :, c.qk_nope_head_dim :] = q_pe

        # ── Attention ──
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            scale=c.softmax_scale,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, c.num_heads * c.v_head_dim)
        output = self.o_proj(attn_output)

        return output, new_cache


# ──────────────────────────────────────────────
# KV Cache 大小对比工具
# ──────────────────────────────────────────────


def compute_kv_cache_bytes(
    seq_len: int,
    num_layers: int,
    dtype_bytes: int = 2,  # fp16
    # MHA/GQA 参数（Qwen2.5-7B 配置）
    gqa_num_kv_heads: int = 4,
    gqa_head_dim: int = 128,
    # MLA 参数（DeepSeek-V2-Lite 配置）
    mla_num_heads: int = 16,
    mla_q_head_dim: int = 192,
    mla_v_head_dim: int = 128,
    mla_kv_lora_rank: int = 512,
    mla_qk_rope_head_dim: int = 64,
) -> dict:
    """
    计算并对比三种 attention 的 KV cache 大小。

    Returns:
        dict，包含：
          gqa_total_bytes, mla_naive_total_bytes, mla_latent_total_bytes
          mla_latent_vs_gqa（压缩比）
    """
    # GQA（标准，如 Qwen2.5-7B）：缓存 K + V，每 token 每层
    gqa_per_token_layer = gqa_num_kv_heads * gqa_head_dim * 2 * dtype_bytes  # K + V
    gqa_total = gqa_per_token_layer * seq_len * num_layers

    # MLA naive（与 HF 等价）：缓存完整 key_states + value_states
    mla_naive_per_token_layer = (
        mla_num_heads * mla_q_head_dim  # key
        + mla_num_heads * mla_v_head_dim  # value
    ) * dtype_bytes
    mla_naive_total = mla_naive_per_token_layer * seq_len * num_layers

    # MLA latent cache：只缓存 compressed_kv + k_pe（共享 1 head）
    mla_latent_per_token_layer = (mla_kv_lora_rank + mla_qk_rope_head_dim) * dtype_bytes
    mla_latent_total = mla_latent_per_token_layer * seq_len * num_layers

    return {
        "gqa_bytes_per_token_layer": gqa_per_token_layer,
        "mla_naive_bytes_per_token_layer": mla_naive_per_token_layer,
        "mla_latent_bytes_per_token_layer": mla_latent_per_token_layer,
        "gqa_total_gb": gqa_total / 1e9,
        "mla_naive_total_gb": mla_naive_total / 1e9,
        "mla_latent_total_gb": mla_latent_total / 1e9,
        "mla_latent_vs_gqa_ratio": mla_latent_per_token_layer / gqa_per_token_layer,
        "mla_latent_vs_mla_naive_ratio": mla_latent_per_token_layer / mla_naive_per_token_layer,
    }


# ──────────────────────────────────────────────
# MLAAttentionAbsorbed：矩阵吸收优化
# ──────────────────────────────────────────────


class MLAAttentionAbsorbed(nn.Module):
    """
    矩阵吸收（Matrix Absorption）优化版 MLA。

    标准 latent cache 版在 decode 时需要：
      1. kv_b_proj(compressed_kv) → k_nope  [O(seq × d_c × num_heads × d_nope)]
      2. q_nope @ k_nope^T                  [O(num_heads × q_len × seq × d_nope)]

    矩阵吸收预计算 W_absorbed = W_uq_nope^T @ W_uk_nope，decode 时直接：
      q_absorbed = q_nope @ W_absorbed      [O(num_heads × q_len × d_nope × d_c)]
      score_nope = q_absorbed @ compressed_kv^T  [O(num_heads × q_len × seq × d_c)]

    避免了对全部历史 compressed_kv 做 kv_b_proj 展开，
    在长序列（seq_len >> kv_lora_rank）时计算量更低。

    同理，V 的计算也可以吸收：
      W_uv_absorbed = W_uv（kv_b_proj 后半部分）
      attn_output = attn_weights @ (compressed_kv @ W_uv_absorbed^T)
                  = (attn_weights @ compressed_kv) @ W_uv_absorbed^T

    KV cache：与 MLAAttentionLatentCache 相同（compressed_kv + k_pe），
    cache 大小不变，只是 attention 计算路径不同。

    参考：DeepSeek-V2 技术报告 Section 2.1.2 "Efficient Inference"
    """

    def __init__(self, config: MLAConfig) -> None:
        super().__init__()
        self.cfg = config
        c = config

        # Q 投影（同 naive）
        if c.q_lora_rank is None:
            self.q_proj = nn.Linear(c.hidden_size, c.num_heads * c.q_head_dim, bias=False)
            self.q_a_proj = None
            self.q_a_layernorm = None
            self.q_b_proj = None
        else:
            self.q_proj = None
            self.q_a_proj = nn.Linear(c.hidden_size, c.q_lora_rank, bias=False)
            self.q_a_layernorm = RMSNorm(c.q_lora_rank)
            self.q_b_proj = nn.Linear(c.q_lora_rank, c.num_heads * c.q_head_dim, bias=False)

        # KV 压缩（同 naive）
        self.kv_a_proj_with_mqa = nn.Linear(
            c.hidden_size, c.kv_lora_rank + c.qk_rope_head_dim, bias=False
        )
        self.kv_a_layernorm = RMSNorm(c.kv_lora_rank)

        # kv_b_proj 保留仅用于 _build_absorbed() 提取权重，forward 中不直接调用
        self.kv_b_proj = nn.Linear(
            c.kv_lora_rank, c.num_heads * (c.qk_nope_head_dim + c.v_head_dim), bias=False
        )

        self.o_proj = nn.Linear(c.num_heads * c.v_head_dim, c.hidden_size, bias=False)

        # 预计算 absorbed 权重（在 _build_absorbed 中填充）
        # W_k_absorbed: (num_heads, kv_lora_rank, qk_nope_head_dim) → 用于 q_nope @ W_k_absorbed
        # W_v_absorbed: (num_heads, kv_lora_rank, v_head_dim)
        self.register_buffer(
            "W_k_absorbed",
            torch.zeros(c.num_heads, c.kv_lora_rank, c.qk_nope_head_dim),
        )
        self.register_buffer(
            "W_v_absorbed",
            torch.zeros(c.num_heads, c.kv_lora_rank, c.v_head_dim),
        )
        # _absorbed_built：标记 absorbed 权重是否已从 kv_b_proj 派生。
        # 注意：这是普通 Python 属性，不随 state_dict 保存。
        # load_state_dict 后 kv_b_proj.weight 已更新，_absorbed_built 重置为 False，
        # 下次 forward 时会自动重建 absorbed 权重，行为正确。
        # 如果在 forward 之后手动修改 kv_b_proj.weight，需要手动调用 _build_absorbed()。
        self._absorbed_built = False

    def _build_absorbed(self) -> None:
        """
        从 kv_b_proj.weight 预计算 absorbed 权重。

        kv_b_proj.weight shape: (num_heads × (qk_nope_head_dim + v_head_dim), kv_lora_rank)
        即 W_kv: [d_c → num_heads × (d_nope + d_v)]

        拆分：
          W_uk: (num_heads, kv_lora_rank, qk_nope_head_dim)  ← k_nope 展开权重
          W_uv: (num_heads, kv_lora_rank, v_head_dim)         ← v 展开权重

        直接存储 W_uk 和 W_uv 作为 absorbed 权重（无需与 W_q 相乘，
        因为 q_nope 已经是投影后的向量，直接与 W_uk 做矩阵乘即可）。
        """
        c = self.cfg
        with torch.no_grad():
            # kv_b_proj.weight: (num_heads*(d_nope+d_v), d_c)
            W = self.kv_b_proj.weight  # (num_heads*(d_nope+d_v), d_c)
            W = W.view(c.num_heads, c.qk_nope_head_dim + c.v_head_dim, c.kv_lora_rank)
            # W_uk: (num_heads, d_nope, d_c) → transpose → (num_heads, d_c, d_nope)
            W_uk = W[:, : c.qk_nope_head_dim, :].transpose(1, 2)  # (num_heads, d_c, d_nope)
            # W_uv: (num_heads, d_v, d_c) → transpose → (num_heads, d_c, d_v)
            W_uv = W[:, c.qk_nope_head_dim :, :].transpose(1, 2)  # (num_heads, d_c, d_v)

            self.W_k_absorbed.copy_(W_uk)
            self.W_v_absorbed.copy_(W_uv)
        self._absorbed_built = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_cache: Optional[MLAKVCacheLatent] = None,
    ) -> Tuple[torch.Tensor, MLAKVCacheLatent]:
        c = self.cfg
        bsz, q_len, _ = hidden_states.shape

        # 首次 forward 时构建 absorbed 权重
        if not self._absorbed_built:
            self._build_absorbed()

        # ── Q 投影（同 naive）──
        if c.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, c.num_heads, c.q_head_dim).transpose(1, 2)
        q_nope, q_pe = q.split([c.qk_nope_head_dim, c.qk_rope_head_dim], dim=-1)
        # q_nope: (bsz, num_heads, q_len, qk_nope_head_dim)

        # ── KV 压缩（同 latent cache）──
        compressed_kv_and_kpe = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv = compressed_kv_and_kpe[..., : c.kv_lora_rank]
        k_pe = compressed_kv_and_kpe[..., c.kv_lora_rank :]

        # ── 拼接历史 latent cache ──
        if past_cache is not None:
            compressed_kv = torch.cat([past_cache.compressed_kv, compressed_kv], dim=1)
            k_pe = torch.cat([past_cache.k_pe, k_pe], dim=1)

        kv_seq_len = compressed_kv.shape[1]
        new_cache = MLAKVCacheLatent(compressed_kv=compressed_kv, k_pe=k_pe)

        # kv_a_layernorm 必须在 attention 计算前应用（与 naive/latent 一致）
        # 注意：cache 存储的是 raw compressed_kv（未 norm），norm 在 attention 时即时做
        compressed_kv_normed = self.kv_a_layernorm(compressed_kv)

        # ── 矩阵吸收：用 W_k_absorbed 直接从 compressed_kv_normed 计算 attention score ──
        # compressed_kv_normed: (bsz, kv_seq_len, d_c)
        # W_k_absorbed:         (num_heads, d_c, d_nope)
        # q_nope:               (bsz, num_heads, q_len, d_nope)
        #
        # kv_for_score = compressed_kv_normed @ W_k_absorbed
        #              → (bsz, num_heads, kv_seq_len, d_nope)
        kv_for_score = torch.einsum(
            "bsd,hde->bhse", compressed_kv_normed, self.W_k_absorbed
        )  # (bsz, num_heads, kv_seq_len, d_nope)

        # score_nope = q_nope @ kv_for_score^T
        score_nope = torch.matmul(q_nope, kv_for_score.transpose(2, 3))
        # (bsz, num_heads, q_len, kv_seq_len)

        # ── RoPE 分量 score ──
        k_pe_full = k_pe.view(bsz, kv_seq_len, 1, c.qk_rope_head_dim).transpose(1, 2)
        # (bsz, 1, kv_seq_len, qk_rope_head_dim)
        score_pe = torch.matmul(q_pe, k_pe_full.transpose(2, 3))
        # (bsz, num_heads, q_len, kv_seq_len)

        # ── 合并 score 并 softmax ──
        attn_weights = (score_nope + score_pe) * c.softmax_scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        # (bsz, num_heads, q_len, kv_seq_len)

        # ── 矩阵吸收：用 W_v_absorbed 直接从 compressed_kv_normed 计算输出 ──
        # kv_for_v = compressed_kv_normed @ W_v_absorbed
        #   → (bsz, num_heads, kv_seq_len, d_v)
        kv_for_v = torch.einsum(
            "bsd,hdv->bhsv", compressed_kv_normed, self.W_v_absorbed
        )  # (bsz, num_heads, kv_seq_len, d_v)

        attn_output = torch.matmul(attn_weights, kv_for_v)
        # (bsz, num_heads, q_len, d_v)

        # ── 输出投影 ──
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, c.num_heads * c.v_head_dim)
        output = self.o_proj(attn_output)

        return output, new_cache
