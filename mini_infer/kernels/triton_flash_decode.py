"""mini_infer/triton_flash_decode.py

Phase 12.5：Flash Decoding（Split-K Attention）Triton 实现。

提供两个公开接口：
  - flash_decode_triton(q, k, v, num_splits=None, scale=None) — split-K 两阶段路径
  - auto_num_splits(seq_len, num_q_heads, batch)              — 自动选 num_splits 的辅助函数

设计约束：
  - 仅支持 decode（query length = 1）
  - KV 为 dense tensor，不走 block_table（实验用，与 triton_attn.py 保持相同接口口径）
  - 支持 GQA（num_q_heads 可 ≠ num_kv_heads，需整除）
  - BLOCK_N=64, HEAD_DIM=128（constexpr）
  - 第一阶段：Triton split kernel，grid=(batch, num_q_heads, num_splits)
  - 第二阶段：PyTorch 归约（在小 num_splits 下与 Triton 归约开销相当，可读性更高）

拓扑关系：
  - 基于 Phase 6.5 triton_attn.py 的 online softmax 框架扩展
  - 不修改 attention.py / model_runner.py，不接入推理主路径（实验性 kernel）
"""

import math

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Phase 1：Split Kernel（每个 program 处理 KV 序列的一个片段）
# ---------------------------------------------------------------------------

@triton.jit
def _flash_decode_split_kernel(
    # 输入指针
    Q_ptr, K_ptr, V_ptr,
    # 输出指针
    PartialOut_ptr, PartialLse_ptr,
    # Q strides: (batch, 1, num_q_heads, head_dim)
    stride_qb, stride_qh, stride_qd,
    # K strides: (batch, seq_len, num_kv_heads, head_dim)
    stride_kb, stride_kn, stride_kh, stride_kd,
    # V strides: same layout as K
    stride_vb, stride_vn, stride_vh, stride_vd,
    # PartialOut strides: (batch, num_splits, num_q_heads, head_dim)
    stride_pob, stride_pos, stride_poh, stride_pod,
    # PartialLse strides: (batch, num_splits, num_q_heads)
    stride_plb, stride_pls, stride_plh,
    # 运行时标量
    seq_len,
    kv_per_split,     # = ceil(seq_len / num_splits)，每个 split 处理的 KV token 数
    num_kv_heads,
    num_q_heads,
    scale,
    # 编译期常量
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """
    Grid: (batch_size, num_q_heads, num_splits)
    每个 program 处理 KV[split_start : split_end] 的 online softmax，
    输出 partial_out（V 加权均值，float16）和 partial_lse（log-sum-exp，float32）。
    """
    batch_idx  = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    split_idx  = tl.program_id(2)

    # GQA：将 q_head 映射到对应的 kv_head
    kv_head_idx = q_head_idx * num_kv_heads // num_q_heads

    # 计算本 split 的 KV 范围
    split_start = split_idx * kv_per_split
    split_end   = tl.minimum(split_start + kv_per_split, seq_len)

    # 加载 Q 向量: (HEAD_DIM,)
    d_range = tl.arange(0, HEAD_DIM)
    q_ptr = Q_ptr + batch_idx * stride_qb + q_head_idx * stride_qh
    q = tl.load(q_ptr + d_range * stride_qd).to(tl.float32)

    # Online softmax 状态
    m_i = tl.full([1], -1e38, dtype=tl.float32)   # 当前最大 score
    l_i = tl.zeros([1], dtype=tl.float32)           # 归一化分母
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)    # 加权 V 累积

    # K/V 在 (batch, kv_head) 维度的基指针
    k_base = K_ptr + batch_idx * stride_kb + kv_head_idx * stride_kh
    v_base = V_ptr + batch_idx * stride_vb + kv_head_idx * stride_vh

    # 遍历本 split 范围内的 KV blocks
    # 若 split_start >= seq_len（空 split），range 直接为空，循环体不执行
    for block_start in range(split_start, split_end, BLOCK_N):
        block_range = block_start + tl.arange(0, BLOCK_N)
        mask = block_range < split_end

        # 加载 K block: (BLOCK_N, HEAD_DIM)
        k_ptrs  = k_base + block_range[:, None] * stride_kn + d_range[None, :] * stride_kd
        k_block = tl.load(k_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)

        # 注意力得分: (BLOCK_N,)
        scores = tl.sum(q[None, :] * k_block, axis=1) * scale
        scores = tl.where(mask, scores, -1e38)

        # Online softmax 更新
        block_max = tl.max(scores[None, :], axis=1)                         # [1]
        m_new     = tl.maximum(m_i, block_max)                              # [1]
        exp_scores = tl.exp(scores - m_new)                                 # (BLOCK_N,)
        l_new     = l_i * tl.exp(m_i - m_new) + tl.sum(exp_scores[None, :], axis=1)  # [1]
        acc       = acc * tl.exp(m_i - m_new)

        # 加载 V block 并累积
        v_ptrs  = v_base + block_range[:, None] * stride_vn + d_range[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=mask[:, None], other=0.0).to(tl.float32)
        acc     = acc + tl.sum(exp_scores[:, None] * v_block, axis=0)

        m_i = m_new
        l_i = l_new

    # partial_lse = m + log(l)；空 split（l==0）输出 -1e38，在归约时权重接近 0
    lse = m_i + tl.log(l_i)
    lse = tl.where(l_i > 0.0, lse, tl.full([1], -1e38, dtype=tl.float32))

    # partial_out = acc / l（V 加权均值）；空 split 输出 0
    safe_l     = tl.where(l_i > 0.0, l_i, tl.full([1], 1.0, dtype=tl.float32))
    partial_out = acc / safe_l                                               # (HEAD_DIM,)

    # 写回 partial_out: (batch, num_splits, num_q_heads, head_dim)
    po_base = (PartialOut_ptr
               + batch_idx  * stride_pob
               + split_idx  * stride_pos
               + q_head_idx * stride_poh)
    tl.store(po_base + d_range * stride_pod, partial_out.to(tl.float16))

    # 写回 partial_lse: (batch, num_splits, num_q_heads) — 单个 float32 标量
    # 用 tl.arange(0, 1) 将标量指针转为 [1] block pointer，与 [1] 的 lse 匹配
    pl_base = (PartialLse_ptr
               + batch_idx  * stride_plb
               + split_idx  * stride_pls
               + q_head_idx * stride_plh)
    tl.store(pl_base + tl.arange(0, 1), lse)


# ---------------------------------------------------------------------------
# Phase 2：PyTorch 归约（数值稳定，在 num_splits <= 32 时开销 < 0.1ms）
# ---------------------------------------------------------------------------

def _reduce_splits(
    partial_out: torch.Tensor,   # (batch, num_splits, num_q_heads, head_dim), float16
    partial_lse: torch.Tensor,   # (batch, num_splits, num_q_heads), float32
) -> torch.Tensor:
    """
    在 num_splits 维度上做数值稳定的 softmax 加权归约。

    数学推导：
      global_lse = logsumexp_s(lse_s)
      w_s = exp(lse_s - global_lse)          # 权重之和 = 1
      out = sum_s(w_s * partial_out_s)
    """
    # max_lse: (batch, 1, num_q_heads)，数值稳定用
    max_lse  = partial_lse.max(dim=1, keepdim=True).values
    exp_lse  = torch.exp(partial_lse - max_lse)                           # (batch, num_splits, num_q_heads)
    weights  = exp_lse / exp_lse.sum(dim=1, keepdim=True)                # 归一化权重
    # (batch, num_splits, num_q_heads, 1) * (batch, num_splits, num_q_heads, head_dim)
    out = (weights.unsqueeze(-1) * partial_out.float()).sum(dim=1)       # (batch, num_q_heads, head_dim)
    return out.to(torch.float16).unsqueeze(1)                            # (batch, 1, num_q_heads, head_dim)


# ---------------------------------------------------------------------------
# 公开接口
# ---------------------------------------------------------------------------

def auto_num_splits(seq_len: int, num_q_heads: int, batch: int = 1, sm_count: int = 128) -> int:
    """
    根据序列长度和 GPU SM 数量自动选择 num_splits。

    策略：
      1. 每个 split 至少需要 BLOCK_N=64 个 KV token，避免空转
      2. 目标使总 thread block 数（batch × num_q_heads × num_splits）接近 sm_count
    """
    BLOCK_N = 64
    max_by_seqlen = max(1, seq_len // BLOCK_N)
    ideal_splits  = max(1, math.ceil(sm_count / (batch * num_q_heads)))
    return min(max_by_seqlen, ideal_splits)


def flash_decode_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_splits: int | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """
    Flash Decoding（split-K）Triton 实现，适用于长序列 decode 场景。

    Args:
        q:          (batch, 1, num_q_heads, head_dim)，float16，CUDA
        k:          (batch, seq_len, num_kv_heads, head_dim)，float16，CUDA
        v:          (batch, seq_len, num_kv_heads, head_dim)，float16，CUDA
        num_splits: KV 序列切分数，None 时自动推算（基于 seq_len 和 SM 数量）
        scale:      注意力缩放因子，默认 head_dim ** -0.5

    Returns:
        out: (batch, 1, num_q_heads, head_dim)，float16，CUDA
    """
    assert q.shape[1] == 1, "flash_decode_triton 仅支持 query length = 1（decode 场景）"
    batch, _, num_q_heads, head_dim = q.shape
    _, seq_len, num_kv_heads, _     = k.shape
    assert seq_len > 0, "seq_len 必须 > 0"
    assert head_dim == 128, f"HEAD_DIM 固定为 128，收到 {head_dim}"
    assert num_q_heads % num_kv_heads == 0, "num_q_heads 必须整除 num_kv_heads"
    assert q.is_cuda and k.is_cuda and v.is_cuda, "所有 tensor 必须在 CUDA 上"

    if scale is None:
        scale = head_dim ** -0.5

    if num_splits is None:
        num_splits = auto_num_splits(seq_len, num_q_heads, batch)

    kv_per_split = math.ceil(seq_len / num_splits)

    # 分配中间 buffer（float16 partial_out，float32 partial_lse）
    partial_out = torch.empty(batch, num_splits, num_q_heads, head_dim,
                              dtype=torch.float16, device=q.device)
    partial_lse = torch.empty(batch, num_splits, num_q_heads,
                              dtype=torch.float32, device=q.device)

    grid = (batch, num_q_heads, num_splits)
    _flash_decode_split_kernel[grid](
        q, k, v, partial_out, partial_lse,
        # Q strides
        q.stride(0), q.stride(2), q.stride(3),
        # K strides
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # V strides
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # PartialOut strides
        partial_out.stride(0), partial_out.stride(1),
        partial_out.stride(2), partial_out.stride(3),
        # PartialLse strides
        partial_lse.stride(0), partial_lse.stride(1), partial_lse.stride(2),
        # Scalars
        seq_len, kv_per_split, num_kv_heads, num_q_heads, scale,
        # Constexpr
        BLOCK_N=64,
        HEAD_DIM=128,
    )

    return _reduce_splits(partial_out, partial_lse)
