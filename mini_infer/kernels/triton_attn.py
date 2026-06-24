"""mini_infer/triton_attn.py

Phase 6.5：Triton decode attention kernel 实验实现。

提供三个公开接口：
  - triton_decode_attention(q, k, v, scale=None) — Triton kernel 路径（GPU）
  - reference_decode_attention(q, k, v)          — PyTorch 参考实现（数值对比用）
  - flash_decode_attention(q, k, v, scale=None)  — flash_attn_with_kvcache 路径（对照组）

设计约束：
  - 仅支持 decode（query length = 1）
  - KV 为 dense tensor，不走 block_table（实验用，与 attention.py 的 PagedDecodeContext 解耦）
  - 支持 GQA（num_q_heads 可 ≠ num_kv_heads，需整除）
  - BLOCK_N=64, HEAD_DIM=128（constexpr，编译期固定）
  - 内部计算精度 float32，输入输出 float16
"""

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton JIT Kernel
# ---------------------------------------------------------------------------

@triton.jit
def _decode_attn_kernel(
    # 指针
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    # Q strides: (batch, 1, num_q_heads, head_dim)
    stride_qb, stride_qh, stride_qd,
    # K strides: (batch, seq_len, num_kv_heads, head_dim)
    stride_kb, stride_kn, stride_kh, stride_kd,
    # V strides: (batch, seq_len, num_kv_heads, head_dim)
    stride_vb, stride_vn, stride_vh, stride_vd,
    # Out strides: (batch, 1, num_q_heads, head_dim)
    stride_ob, stride_oh, stride_od,
    # 运行时标量
    seq_len,
    num_kv_heads,
    num_q_heads,
    scale,
    # 编译期常量
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Grid: (batch_size, num_q_heads)
    每个 program 处理一个 (batch_idx, q_head_idx) 对。
    使用 online softmax（Milakov & Gimelshein 2018）迭代 KV blocks。
    """
    batch_idx  = tl.program_id(0)
    q_head_idx = tl.program_id(1)

    # GQA：将 q_head 映射到对应的 kv_head
    kv_head_idx = q_head_idx * num_kv_heads // num_q_heads

    # 加载 Q 向量：(HEAD_DIM,)
    d_range = tl.arange(0, HEAD_DIM)
    q_ptr = Q_ptr + batch_idx * stride_qb + q_head_idx * stride_qh
    q = tl.load(q_ptr + d_range * stride_qd).to(tl.float32)

    # Online softmax 状态初始化
    m_i = tl.full([1], -1e38, dtype=tl.float32)   # 当前最大值
    l_i = tl.zeros([1], dtype=tl.float32)           # 归一化分母
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)    # 加权累积 V

    # K/V 在 (batch, kv_head) 维度的基指针
    k_base = K_ptr + batch_idx * stride_kb + kv_head_idx * stride_kh
    v_base = V_ptr + batch_idx * stride_vb + kv_head_idx * stride_vh

    # 分 block 迭代整个 KV 序列
    for block_start in range(0, seq_len, BLOCK_N):
        block_range = block_start + tl.arange(0, BLOCK_N)  # (BLOCK_N,)
        mask = block_range < seq_len

        # 加载 K block: (BLOCK_N, HEAD_DIM)
        k_ptrs  = k_base + block_range[:, None] * stride_kn + d_range[None, :] * stride_kd
        k_block = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

        # 注意力得分: q (HEAD_DIM,) · k_block[i] (HEAD_DIM,) → scores (BLOCK_N,)
        scores = tl.sum(q[None, :] * k_block, axis=1) * scale
        scores = tl.where(mask, scores, -1e38)   # padding 位置屏蔽

        # Online softmax 更新
        # 使用 scores[None, :] 升维后 reduce，结果为 [1]，与 m_i 的 blocked encoding 一致
        # 避免 tl.max(scores, 0) 返回 scalar 导致 encoding 不匹配的编译报错
        block_max  = tl.max(scores[None, :], axis=1)                       # [1]
        m_new      = tl.maximum(m_i, block_max)                            # [1]
        exp_scores = tl.exp(scores - m_new)                                # (BLOCK_N,)
        l_new      = l_i * tl.exp(m_i - m_new) + tl.sum(exp_scores[None, :], axis=1)  # [1]

        # 重新缩放已有 accumulator
        acc = acc * tl.exp(m_i - m_new)

        # 加载 V block 并累积: exp_scores (BLOCK_N,) × V (BLOCK_N, HEAD_DIM) → (HEAD_DIM,)
        v_ptrs  = v_base + block_range[:, None] * stride_vn + d_range[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
        acc     = acc + tl.sum(exp_scores[:, None] * v_block, axis=0)

        m_i = m_new
        l_i = l_new

    # 归一化并写回
    out_vec = acc / l_i
    out_ptr = Out_ptr + batch_idx * stride_ob + q_head_idx * stride_oh
    tl.store(out_ptr + d_range * stride_od, out_vec.to(tl.float16))


# ---------------------------------------------------------------------------
# Python Wrappers
# ---------------------------------------------------------------------------

def triton_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Triton decode attention，query length = 1，dense KV。

    Args:
        q:     (batch, 1, num_q_heads, head_dim)，float16，CUDA
        k:     (batch, seq_len, num_kv_heads, head_dim)，float16，CUDA
        v:     (batch, seq_len, num_kv_heads, head_dim)，float16，CUDA
        scale: 注意力缩放因子，默认 head_dim ** -0.5

    Returns:
        out:   (batch, 1, num_q_heads, head_dim)，float16，CUDA
    """
    assert q.shape[1] == 1, "triton_decode_attention only supports query length = 1"
    batch, _, num_q_heads, head_dim = q.shape
    _, seq_len, num_kv_heads, _ = k.shape
    assert head_dim == 128, f"HEAD_DIM 固定为 128，收到 {head_dim}"
    assert num_q_heads % num_kv_heads == 0, "num_q_heads 必须整除 num_kv_heads"
    assert q.is_cuda and k.is_cuda and v.is_cuda

    if scale is None:
        scale = head_dim ** -0.5

    out  = torch.empty_like(q)
    grid = (batch, num_q_heads)

    _decode_attn_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(2), out.stride(3),
        seq_len, num_kv_heads, num_q_heads, scale,
        BLOCK_N=64,
        HEAD_DIM=128,
    )
    return out


def reference_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """
    PyTorch 参考实现，用于数值正确性对比。

    Args:
        q:     (batch, 1, num_q_heads, head_dim)
        k:     (batch, seq_len, num_kv_heads, head_dim)
        v:     (batch, seq_len, num_kv_heads, head_dim)
        scale: 默认 head_dim ** -0.5

    Returns:
        out:   (batch, 1, num_q_heads, head_dim)，与输入同 dtype
    """
    batch, _, num_q_heads, head_dim = q.shape
    _, seq_len, num_kv_heads, _ = k.shape
    groups = num_q_heads // num_kv_heads

    if scale is None:
        scale = head_dim ** -0.5

    # GQA 展开: (batch, seq_len, num_kv_heads, head_dim) → (batch, seq_len, num_q_heads, head_dim)
    k_exp = k.repeat_interleave(groups, dim=2)
    v_exp = v.repeat_interleave(groups, dim=2)

    q_t   = q.permute(0, 2, 1, 3).float()        # (batch, num_q_heads, 1, head_dim)
    k_t   = k_exp.permute(0, 2, 3, 1).float()    # (batch, num_q_heads, head_dim, seq_len)
    v_t   = v_exp.permute(0, 2, 1, 3).float()    # (batch, num_q_heads, seq_len, head_dim)

    scores = torch.matmul(q_t, k_t) * scale      # (batch, num_q_heads, 1, seq_len)
    attn   = torch.softmax(scores, dim=-1)
    out_t  = torch.matmul(attn, v_t)             # (batch, num_q_heads, 1, head_dim)

    return out_t.permute(0, 2, 1, 3).to(q.dtype) # (batch, 1, num_q_heads, head_dim)


def flash_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """
    flash_attn_with_kvcache 路径，作为 benchmark 对照组。

    Args / Returns: 同 triton_decode_attention。
    """
    from flash_attn import flash_attn_with_kvcache

    if scale is None:
        scale = q.shape[-1] ** -0.5

    # flash_attn_with_kvcache 接受 (batch, seqlen, nheads, d_head)
    # 不传 cache_seqlens → 使用 k_cache/v_cache 的全部长度，causal=False
    return flash_attn_with_kvcache(
        q, k, v,
        softmax_scale=scale,
        causal=False,
    )
