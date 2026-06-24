"""tests/test_flash_decode.py

Phase 12.5：Flash Decoding（split-K）正确性测试。

覆盖范围：
  - num_splits=1 时退化为标准 online softmax，与 reference 差异 < 1e-3
  - 各 (seq_len, num_splits) 组合，max_diff < 1e-2 vs reference_decode_attention
  - 各组合，max_diff < 1e-2 vs flash_attn_with_kvcache（对照组）
  - GQA 场景（num_q_heads != num_kv_heads）
  - 空 split 边界情况（seq_len < num_splits * BLOCK_N）
  - auto_num_splits 输出合理范围
"""

import pytest
import torch


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def make_qkv(batch, seq_len, num_q_heads, num_kv_heads, head_dim=128, device="cuda"):
    torch.manual_seed(42)
    q = torch.randn(batch, 1,       num_q_heads, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=torch.float16, device=device)
    return q, k, v


# ---------------------------------------------------------------------------
# 无 GPU 测试（import / shape / auto_num_splits）
# ---------------------------------------------------------------------------

def test_import():
    """模块可以正常导入。"""
    from mini_infer.kernels.triton_flash_decode import flash_decode_triton, auto_num_splits  # noqa: F401


def test_auto_num_splits_basic():
    """auto_num_splits 在合理范围内。"""
    from mini_infer.kernels.triton_flash_decode import auto_num_splits

    # seq_len=64：每 split 最少 BLOCK_N=64 tokens → max_splits=1
    assert auto_num_splits(64, 12, batch=1) == 1

    # seq_len=4096，1.5B（12 heads）：应该选 > 1
    ns = auto_num_splits(4096, 12, batch=1)
    assert 1 <= ns <= 64, f"got {ns}"

    # num_splits >= 1 对所有输入
    for seq in [128, 512, 1024, 2048, 4096]:
        for nh in [2, 12, 28]:
            ns = auto_num_splits(seq, nh, batch=1)
            assert ns >= 1


# ---------------------------------------------------------------------------
# GPU 测试
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_output_shape():
    """输出 shape 与 q 相同。"""
    from mini_infer.kernels.triton_flash_decode import flash_decode_triton

    q, k, v = make_qkv(2, 256, 12, 2)
    out = flash_decode_triton(q, k, v, num_splits=4)
    assert out.shape == q.shape, f"expected {q.shape}, got {out.shape}"
    assert out.dtype == torch.float16


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_num_splits_1_matches_reference():
    """num_splits=1 时，结果与 reference_decode_attention 的差异 < 1e-3。"""
    from mini_infer.kernels.triton_flash_decode import flash_decode_triton
    from mini_infer.kernels.triton_attn import reference_decode_attention

    q, k, v = make_qkv(1, 512, 12, 2)
    out_fd  = flash_decode_triton(q, k, v, num_splits=1)
    out_ref = reference_decode_attention(q, k, v)

    max_diff = (out_fd - out_ref).abs().max().item()
    assert max_diff < 1e-3, f"num_splits=1 max_diff={max_diff:.2e} >= 1e-3"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
@pytest.mark.parametrize("seq_len,num_splits", [
    (128,  2),
    (256,  4),
    (512,  4),
    (512,  8),
    (1024, 8),
    (2048, 16),
    (4096, 16),
    (4096, 32),
])
def test_correctness_vs_reference(seq_len, num_splits):
    """各 (seq_len, num_splits) 组合，max_diff < 1e-2 vs reference。"""
    from mini_infer.kernels.triton_flash_decode import flash_decode_triton
    from mini_infer.kernels.triton_attn import reference_decode_attention

    q, k, v = make_qkv(1, seq_len, 12, 2)
    out_fd  = flash_decode_triton(q, k, v, num_splits=num_splits)
    out_ref = reference_decode_attention(q, k, v)

    max_diff = (out_fd - out_ref).abs().max().item()
    assert max_diff < 1e-2, f"seq={seq_len} splits={num_splits} max_diff={max_diff:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
@pytest.mark.parametrize("seq_len,num_splits", [
    (256,  4),
    (1024, 8),
    (4096, 16),
])
def test_correctness_vs_flash_attn(seq_len, num_splits):
    """max_diff < 1e-2 vs flash_attn_with_kvcache。"""
    from mini_infer.kernels.triton_flash_decode import flash_decode_triton
    from mini_infer.kernels.triton_attn import flash_decode_attention

    q, k, v = make_qkv(1, seq_len, 12, 2)
    out_fd    = flash_decode_triton(q, k, v, num_splits=num_splits)
    out_flash = flash_decode_attention(q, k, v)

    max_diff = (out_fd - out_flash).abs().max().item()
    assert max_diff < 1e-2, f"vs flash_attn: seq={seq_len} splits={num_splits} max_diff={max_diff:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_gqa_num_q_ne_kv():
    """GQA：num_q_heads(28) != num_kv_heads(4) 时正确。"""
    from mini_infer.kernels.triton_flash_decode import flash_decode_triton
    from mini_infer.kernels.triton_attn import reference_decode_attention

    q, k, v = make_qkv(1, 512, 28, 4)
    out_fd  = flash_decode_triton(q, k, v, num_splits=4)
    out_ref = reference_decode_attention(q, k, v)

    max_diff = (out_fd - out_ref).abs().max().item()
    assert max_diff < 1e-2, f"GQA max_diff={max_diff:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_batch_size_gt1():
    """batch > 1 时输出正确。"""
    from mini_infer.kernels.triton_flash_decode import flash_decode_triton
    from mini_infer.kernels.triton_attn import reference_decode_attention

    q, k, v = make_qkv(4, 512, 12, 2)
    out_fd  = flash_decode_triton(q, k, v, num_splits=4)
    out_ref = reference_decode_attention(q, k, v)

    max_diff = (out_fd - out_ref).abs().max().item()
    assert max_diff < 1e-2, f"batch=4 max_diff={max_diff:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_kv_per_split_less_than_block_n():
    """kv_per_split < BLOCK_N 时最后一块稀疏 mask 仍正确。"""
    from mini_infer.kernels.triton_flash_decode import flash_decode_triton
    from mini_infer.kernels.triton_attn import reference_decode_attention

    # seq_len=64, num_splits=8 → kv_per_split=8 < BLOCK_N=64，每个循环只有 8 个有效 token
    # 每个 split 都非空，但 block 内大部分 token 被 mask
    q, k, v = make_qkv(1, 64, 12, 2)
    out_fd  = flash_decode_triton(q, k, v, num_splits=8)
    out_ref = reference_decode_attention(q, k, v)

    max_diff = (out_fd - out_ref).abs().max().item()
    assert max_diff < 1e-2, f"kv_per_split<BLOCK_N max_diff={max_diff:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_truly_empty_splits():
    """split_start >= seq_len 的空 split 不影响输出（写 lse=-1e38，权重接近 0）。"""
    from mini_infer.kernels.triton_flash_decode import flash_decode_triton
    from mini_infer.kernels.triton_attn import reference_decode_attention

    # seq_len=64, num_splits=9 → kv_per_split=ceil(64/9)=8
    # split 8: split_start=64, split_end=min(72,64)=64 → 空 split（split_start==split_end）
    q, k, v = make_qkv(1, 64, 12, 2)
    out_fd  = flash_decode_triton(q, k, v, num_splits=9)
    out_ref = reference_decode_attention(q, k, v)

    max_diff = (out_fd - out_ref).abs().max().item()
    assert max_diff < 1e-2, f"empty-splits max_diff={max_diff:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_seq_len_not_divisible_by_block_n():
    """seq_len 非 BLOCK_N(=64) 整除时，最后一个 partial block 的 masking 正确。"""
    from mini_infer.kernels.triton_flash_decode import flash_decode_triton
    from mini_infer.kernels.triton_attn import reference_decode_attention

    for seq_len in [100, 513, 1000]:
        q, k, v = make_qkv(1, seq_len, 12, 2)
        out_fd  = flash_decode_triton(q, k, v, num_splits=4)
        out_ref = reference_decode_attention(q, k, v)
        max_diff = (out_fd - out_ref).abs().max().item()
        assert max_diff < 1e-2, f"seq_len={seq_len} max_diff={max_diff:.3e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 CUDA")
def test_auto_num_splits_integration():
    """使用 auto_num_splits 默认路径（num_splits=None）结果正确。"""
    from mini_infer.kernels.triton_flash_decode import flash_decode_triton
    from mini_infer.kernels.triton_attn import reference_decode_attention

    q, k, v = make_qkv(1, 2048, 12, 2)
    out_fd  = flash_decode_triton(q, k, v)          # num_splits=None → 自动
    out_ref = reference_decode_attention(q, k, v)

    max_diff = (out_fd - out_ref).abs().max().item()
    assert max_diff < 1e-2, f"auto splits max_diff={max_diff:.3e}"
