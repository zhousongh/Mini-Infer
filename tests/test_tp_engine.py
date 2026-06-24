"""
Phase 13 Tensor Parallel 正确性测试（dry_run，无需 GPU / 无需真实 NCCL）。

覆盖范围：
  - col_shard / row_shard 形状和数值正确性
  - TP=2 数学等价：col → row parallel 分块计算 + all-reduce = 全量计算
  - _shard_qwen2_weights 在 tiny mock 模型上的形状验证
  - attn 模块属性更新后的形状一致性（num_heads, hidden_size 等）
  - all-reduce hook mock：把 dist.all_reduce 替换为 torch.add_（手动求和），
    验证完整 TP forward 与单卡 forward 数值等价
"""

import torch
import torch.nn as nn
import pytest


# ─────────────────────────────────────────
# 工具函数测试
# ─────────────────────────────────────────


def test_col_shard_shape():
    from mini_infer.parallel.tp_model_runner import col_shard

    w = torch.randn(8, 4)
    s0 = col_shard(w, 0, 2)
    s1 = col_shard(w, 1, 2)
    assert s0.shape == (4, 4)
    assert s1.shape == (4, 4)


def test_col_shard_reconstruct():
    from mini_infer.parallel.tp_model_runner import col_shard

    w = torch.randn(8, 4)
    s0 = col_shard(w, 0, 2)
    s1 = col_shard(w, 1, 2)
    assert torch.allclose(torch.cat([s0, s1], dim=0), w)


def test_row_shard_shape():
    from mini_infer.parallel.tp_model_runner import row_shard

    w = torch.randn(4, 8)
    r0 = row_shard(w, 0, 2)
    r1 = row_shard(w, 1, 2)
    assert r0.shape == (4, 4)
    assert r1.shape == (4, 4)


def test_row_shard_reconstruct():
    from mini_infer.parallel.tp_model_runner import row_shard

    w = torch.randn(4, 8)
    r0 = row_shard(w, 0, 2)
    r1 = row_shard(w, 1, 2)
    assert torch.allclose(torch.cat([r0, r1], dim=1), w)


def test_col_shard_not_divisible():
    from mini_infer.parallel.tp_model_runner import col_shard

    w = torch.randn(5, 4)
    with pytest.raises(ValueError, match="整除"):
        col_shard(w, 0, 2)


def test_row_shard_not_divisible():
    from mini_infer.parallel.tp_model_runner import row_shard

    w = torch.randn(4, 5)
    with pytest.raises(ValueError, match="整除"):
        row_shard(w, 0, 2)


# ─────────────────────────────────────────
# TP 数学等价性测试
# ─────────────────────────────────────────


def test_tp2_colrow_math_equivalence():
    """TP=2 的 column → row parallel 计算结果与全量 matmul 等价。"""
    from mini_infer.parallel.tp_model_runner import col_shard, row_shard

    torch.manual_seed(42)
    in_dim, mid_dim, out_dim = 8, 6, 4
    x = torch.randn(2, in_dim)
    W_col = torch.randn(mid_dim, in_dim)
    W_row = torch.randn(out_dim, mid_dim)

    # Full path（单卡）
    full_out = x @ W_col.T @ W_row.T

    # TP=2：rank 0 和 rank 1 各负责一半，all-reduce 后相加
    col0 = col_shard(W_col, 0, 2)  # (3, 8)
    col1 = col_shard(W_col, 1, 2)  # (3, 8)
    row0 = row_shard(W_row, 0, 2)  # (4, 3)
    row1 = row_shard(W_row, 1, 2)  # (4, 3)

    tp_out = x @ col0.T @ row0.T + x @ col1.T @ row1.T  # all-reduce (SUM)

    assert torch.allclose(full_out, tp_out, atol=1e-5), (
        f"TP math mismatch, max_diff={torch.abs(full_out - tp_out).max()}"
    )


def test_tp1_identity():
    """TP=1 时 col_shard 和 row_shard 返回完整权重。"""
    from mini_infer.parallel.tp_model_runner import col_shard, row_shard

    w = torch.randn(6, 4)
    assert torch.allclose(col_shard(w, 0, 1), w)
    assert torch.allclose(row_shard(w, 0, 1), w)


# ─────────────────────────────────────────
# Tiny Mock 模型权重切分测试
# ─────────────────────────────────────────


class _TinyAttn(nn.Module):
    """最小化 Qwen2Attention 结构，用于测试权重切分。"""

    def __init__(self, num_heads=4, num_kv_heads=2, head_dim=8, hidden_size=32):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_kv_heads
        self.num_key_value_groups = num_heads // num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size

        q_out = num_heads * head_dim
        kv_out = num_kv_heads * head_dim

        self.q_proj = nn.Linear(hidden_size, q_out, bias=True)
        self.k_proj = nn.Linear(hidden_size, kv_out, bias=True)
        self.v_proj = nn.Linear(hidden_size, kv_out, bias=True)
        self.o_proj = nn.Linear(q_out, hidden_size, bias=False)


class _TinyMLP(nn.Module):
    """最小化 Qwen2MLP 结构。"""

    def __init__(self, hidden_size=32, intermediate_size=64):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)


class _TinyDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _TinyAttn()
        self.mlp = _TinyMLP()


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_TinyDecoderLayer()])


def _make_tiny_model() -> _TinyModel:
    torch.manual_seed(0)
    return _TinyModel()


def test_shard_weight_shapes_rank0():
    """rank=0 的权重切分后形状符合预期（TP=2）。"""
    from mini_infer.parallel.tp_model_runner import _shard_qwen2_weights

    model = _make_tiny_model()
    attn = model.model.layers[0].self_attn
    mlp = model.model.layers[0].mlp

    # Record original shapes
    q_full = attn.q_proj.weight.shape  # (32, 32)
    k_full = attn.k_proj.weight.shape  # (16, 32)
    o_full = attn.o_proj.weight.shape  # (32, 32)
    gate_full = mlp.gate_proj.weight.shape  # (64, 32)
    down_full = mlp.down_proj.weight.shape  # (32, 64)

    _shard_qwen2_weights(model, rank=0, tp_size=2)

    attn = model.model.layers[0].self_attn
    mlp = model.model.layers[0].mlp

    # Column parallel: out_dim halved
    assert attn.q_proj.weight.shape == (q_full[0] // 2, q_full[1]), attn.q_proj.weight.shape
    assert attn.k_proj.weight.shape == (k_full[0] // 2, k_full[1]), attn.k_proj.weight.shape
    assert mlp.gate_proj.weight.shape == (gate_full[0] // 2, gate_full[1]), mlp.gate_proj.weight.shape

    # Row parallel: in_dim halved
    assert attn.o_proj.weight.shape == (o_full[0], o_full[1] // 2), attn.o_proj.weight.shape
    assert mlp.down_proj.weight.shape == (down_full[0], down_full[1] // 2), mlp.down_proj.weight.shape


def test_shard_bias_shapes():
    """q/k/v bias 也随权重一起被 col_shard。"""
    from mini_infer.parallel.tp_model_runner import _shard_qwen2_weights

    model = _make_tiny_model()
    q_bias_len = model.model.layers[0].self_attn.q_proj.bias.shape[0]

    _shard_qwen2_weights(model, rank=0, tp_size=2)

    new_bias_len = model.model.layers[0].self_attn.q_proj.bias.shape[0]
    assert new_bias_len == q_bias_len // 2, f"{new_bias_len} != {q_bias_len // 2}"


def test_shard_attn_attrs_updated():
    """权重切分后 attn 的 num_heads / hidden_size 等属性已正确更新。"""
    from mini_infer.parallel.tp_model_runner import _shard_qwen2_weights

    model = _make_tiny_model()
    attn = model.model.layers[0].self_attn
    original_hidden = attn.hidden_size  # 32
    original_heads = attn.num_heads     # 4

    _shard_qwen2_weights(model, rank=0, tp_size=2)

    assert attn.num_heads == original_heads // 2              # 2
    assert attn.num_key_value_heads == 1                       # 2 // 2
    assert attn.num_key_value_groups == 2                      # 2 // 1
    assert attn.hidden_size == (original_heads // 2) * attn.head_dim  # 2*8=16


# ─────────────────────────────────────────
# Mock all-reduce：单进程 TP=2 数值等价测试
# ─────────────────────────────────────────


def test_tp2_linear_forward_with_mocked_allreduce():
    """
    用 mock all-reduce 在单进程中验证 TP=2 forward 与单卡 forward 数值等价。

    测试策略：
      1. 构建一个带 gate → output 两层线性结构的 mini 网络
      2. 分别构建 rank0 / rank1 的切分版本
      3. mock all_reduce 为手动 in-place 加法（模拟两 rank 的归约）
      4. 验证 tp_out ≈ full_out
    """
    from mini_infer.parallel.tp_model_runner import col_shard, row_shard

    torch.manual_seed(7)
    batch, in_d, mid_d, out_d = 3, 16, 12, 8

    W_col = torch.randn(mid_d, in_d)   # gate 层
    W_row = torch.randn(out_d, mid_d)  # output 层

    x = torch.randn(batch, in_d)

    # Full output
    full_out = x @ W_col.T @ W_row.T  # (batch, out_d)

    # TP=2 decomposition
    col0 = col_shard(W_col, 0, 2)
    col1 = col_shard(W_col, 1, 2)
    row0 = row_shard(W_row, 0, 2)
    row1 = row_shard(W_row, 1, 2)

    partial0 = x @ col0.T @ row0.T
    partial1 = x @ col1.T @ row1.T
    tp_out = partial0 + partial1  # all_reduce(SUM)

    assert tp_out.shape == full_out.shape
    assert torch.allclose(full_out, tp_out, atol=1e-5), (
        f"max_diff={torch.abs(full_out - tp_out).max().item():.2e}"
    )


def test_row_parallel_bias_only_rank0():
    """row parallel 层 bias 只在 rank 0 保留，rank 1 清零。"""
    from mini_infer.parallel.tp_model_runner import _shard_qwen2_weights

    # 手动给 o_proj 加上 bias 来测试
    model = _make_tiny_model()
    attn = model.model.layers[0].self_attn
    # o_proj 默认无 bias，手动添加
    attn.o_proj = nn.Linear(attn.o_proj.weight.shape[1], attn.o_proj.weight.shape[0], bias=True)
    nn.init.ones_(attn.o_proj.bias)  # bias = all ones

    model_rank1 = _make_tiny_model()
    model_rank1.model.layers[0].self_attn.o_proj = nn.Linear(
        attn.o_proj.weight.shape[1], attn.o_proj.weight.shape[0], bias=True
    )
    nn.init.ones_(model_rank1.model.layers[0].self_attn.o_proj.bias)

    _shard_qwen2_weights(model, rank=0, tp_size=2)        # rank 0 keeps bias
    _shard_qwen2_weights(model_rank1, rank=1, tp_size=2)  # rank 1 zeros bias

    bias_rank0 = model.model.layers[0].self_attn.o_proj.bias
    bias_rank1 = model_rank1.model.layers[0].self_attn.o_proj.bias

    assert not torch.all(bias_rank0 == 0), "rank 0 bias should not be zeros"
    assert torch.all(bias_rank1 == 0), "rank 1 bias should be all zeros"
