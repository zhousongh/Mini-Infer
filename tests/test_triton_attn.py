"""tests/test_triton_attn.py

覆盖范围：triton_decode_attention 的数值正确性验证（Phase 6.5 Step 2）。

测试结构：
  - 无 GPU：仅验证函数签名和模块可导入（dry_run 桩）
  - 有 GPU：
    - test_vs_reference: Triton kernel vs PyTorch reference，max_diff < 1e-2（fp16 精度）
    - test_vs_flash:     Triton kernel vs flash_decode_attention，max_diff < 1e-2
    - test_gqa:          GQA 映射（28 Q heads / 4 KV heads，Qwen2.5-7B 配置）
    - test_batch:        batch=8，确认批处理独立性
    - test_long_seq:     seq_len=2048，验证跨多个 BLOCK_N=64 的迭代正确性
"""

import pytest
import torch

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _rand_qkv(batch, seq_len, num_q_heads, num_kv_heads, head_dim, device):
    """生成随机 float16 的 Q/K/V 张量（已归一化，避免数值爆炸）。"""
    q = torch.randn(batch, 1,       num_q_heads,  head_dim, device=device, dtype=torch.float16) * 0.1
    k = torch.randn(batch, seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float16) * 0.1
    v = torch.randn(batch, seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float16) * 0.1
    return q, k, v


# ---------------------------------------------------------------------------
# Dry-run（无 GPU）：仅验证函数存在
# ---------------------------------------------------------------------------

def test_import():
    """无 GPU 可以跑：验证三个公开函数均可导入。"""
    from mini_infer.kernels.triton_attn import (
        triton_decode_attention,
        reference_decode_attention,
        flash_decode_attention,
    )
    assert callable(triton_decode_attention)
    assert callable(reference_decode_attention)
    assert callable(flash_decode_attention)


def test_reference_cpu():
    """无 GPU 可以跑：reference_decode_attention 在 CPU float32 上应能运行。"""
    from mini_infer.kernels.triton_attn import reference_decode_attention

    q = torch.randn(2, 1, 4, 128, dtype=torch.float32)
    k = torch.randn(2, 16, 2, 128, dtype=torch.float32)
    v = torch.randn(2, 16, 2, 128, dtype=torch.float32)
    out = reference_decode_attention(q, k, v)
    assert out.shape == (2, 1, 4, 128)
    assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# GPU 测试（需要 CUDA）
# ---------------------------------------------------------------------------

gpu_only = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="需要 CUDA GPU"
)


@gpu_only
def test_vs_reference_basic():
    """Triton kernel vs PyTorch reference，标准 GQA 配置，max_diff < 1e-2。"""
    from mini_infer.kernels.triton_attn import triton_decode_attention, reference_decode_attention

    batch, seq_len = 2, 128
    num_q_heads, num_kv_heads, head_dim = 28, 4, 128
    device = "cuda"

    q, k, v = _rand_qkv(batch, seq_len, num_q_heads, num_kv_heads, head_dim, device)

    out_triton = triton_decode_attention(q, k, v)
    out_ref    = reference_decode_attention(q, k, v)

    assert out_triton.shape == out_ref.shape, f"shape mismatch: {out_triton.shape} vs {out_ref.shape}"
    max_diff = (out_triton.float() - out_ref.float()).abs().max().item()
    print(f"\n[test_vs_reference_basic] max_diff = {max_diff:.6f}")
    assert max_diff < 1e-2, f"max_diff {max_diff:.4f} 超过 1e-2 阈值"


@gpu_only
def test_vs_flash():
    """Triton kernel vs flash_attn_with_kvcache，max_diff < 1e-2。"""
    from mini_infer.kernels.triton_attn import triton_decode_attention, flash_decode_attention

    batch, seq_len = 2, 256
    num_q_heads, num_kv_heads, head_dim = 28, 4, 128
    device = "cuda"

    q, k, v = _rand_qkv(batch, seq_len, num_q_heads, num_kv_heads, head_dim, device)

    out_triton = triton_decode_attention(q, k, v)
    out_flash  = flash_decode_attention(q, k, v)

    max_diff = (out_triton.float() - out_flash.float()).abs().max().item()
    print(f"\n[test_vs_flash] max_diff = {max_diff:.6f}")
    assert max_diff < 1e-2, f"max_diff {max_diff:.4f} 超过 1e-2 阈值"


@gpu_only
def test_gqa_qwen_config():
    """Qwen2.5-7B 真实配置：28 Q heads / 4 KV heads / head_dim=128。"""
    from mini_infer.kernels.triton_attn import triton_decode_attention, reference_decode_attention

    batch, seq_len = 1, 512
    num_q_heads, num_kv_heads, head_dim = 28, 4, 128
    device = "cuda"

    q, k, v = _rand_qkv(batch, seq_len, num_q_heads, num_kv_heads, head_dim, device)

    out_triton = triton_decode_attention(q, k, v)
    out_ref    = reference_decode_attention(q, k, v)

    max_diff = (out_triton.float() - out_ref.float()).abs().max().item()
    print(f"\n[test_gqa_qwen_config] max_diff = {max_diff:.6f}")
    assert max_diff < 1e-2


@gpu_only
def test_batch8():
    """batch=8，验证批处理独立性（逐 sample 与 batch=1 结果一致）。"""
    from mini_infer.kernels.triton_attn import triton_decode_attention

    batch, seq_len = 8, 128
    num_q_heads, num_kv_heads, head_dim = 28, 4, 128
    device = "cuda"

    q, k, v = _rand_qkv(batch, seq_len, num_q_heads, num_kv_heads, head_dim, device)

    out_batch = triton_decode_attention(q, k, v)

    # 逐 sample 单独计算，结果应与 batch 结果一致
    for i in range(batch):
        out_single = triton_decode_attention(
            q[i:i+1], k[i:i+1], v[i:i+1]
        )
        diff = (out_batch[i:i+1].float() - out_single.float()).abs().max().item()
        assert diff < 1e-4, f"sample {i}: batch vs single diff = {diff}"


@gpu_only
def test_long_seq():
    """seq_len=2048，验证跨 32 个 BLOCK_N=64 块的迭代正确性。"""
    from mini_infer.kernels.triton_attn import triton_decode_attention, reference_decode_attention

    batch, seq_len = 1, 2048
    num_q_heads, num_kv_heads, head_dim = 28, 4, 128
    device = "cuda"

    q, k, v = _rand_qkv(batch, seq_len, num_q_heads, num_kv_heads, head_dim, device)

    out_triton = triton_decode_attention(q, k, v)
    out_ref    = reference_decode_attention(q, k, v)

    max_diff = (out_triton.float() - out_ref.float()).abs().max().item()
    print(f"\n[test_long_seq] seq_len={seq_len}, max_diff = {max_diff:.6f}")
    assert max_diff < 1e-2, f"max_diff {max_diff:.4f} 超过 1e-2 阈值"


@gpu_only
def test_seq_not_multiple_of_block():
    """seq_len=100（非 BLOCK_N=64 整倍数），验证 padding 掩码正确。"""
    from mini_infer.kernels.triton_attn import triton_decode_attention, reference_decode_attention

    batch, seq_len = 2, 100
    num_q_heads, num_kv_heads, head_dim = 4, 4, 128
    device = "cuda"

    q, k, v = _rand_qkv(batch, seq_len, num_q_heads, num_kv_heads, head_dim, device)

    out_triton = triton_decode_attention(q, k, v)
    out_ref    = reference_decode_attention(q, k, v)

    max_diff = (out_triton.float() - out_ref.float()).abs().max().item()
    print(f"\n[test_seq_not_multiple_of_block] seq_len={seq_len}, max_diff = {max_diff:.6f}")
    assert max_diff < 1e-2
