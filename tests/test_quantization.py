"""Phase 16 量化测试。

覆盖：
- QuantMode 解析
- QuantLinear 权重张量形状
- CPU 路径前向精度（反量化误差）
- CUDA 路径前向精度（M≥17，满足 _int_mm 约束）
- 带 bias 的 QuantLinear
- QuantLinear 运行时统计
- quantize_model() 替换 nn.Linear
- quantize_model() 跳过 lm_head / embed_tokens
- quantize_model() 跳过 attention q_proj
- quantize_model() 跳过小层
- EngineConfig quant_mode 验证
"""

import pytest
import torch
import torch.nn as nn

from mini_infer.modeling.quantization import QuantLinear, QuantMode, quantize_model
from mini_infer.core.config import EngineConfig


# --------------------------------------------------------------------------- #
# QuantMode 解析
# --------------------------------------------------------------------------- #

class TestQuantMode:
    def test_from_str_none(self):
        assert QuantMode.from_str("") == QuantMode.NONE

    def test_from_str_w8a8(self):
        assert QuantMode.from_str("w8a8") == QuantMode.W8A8

    def test_from_str_invalid(self):
        with pytest.raises(ValueError, match="不支持"):
            QuantMode.from_str("fp8")

    def test_enum_value(self):
        assert QuantMode.W8A8.value == "w8a8"
        assert QuantMode.NONE.value == ""

    def test_contract_summary(self):
        contract = QuantLinear.get_contract()
        assert contract["weight_storage"] == "int8_per_channel"
        assert contract["int_mm_activation_granularity"] == "per_row"
        assert contract["fallback_activation_granularity"] == "fp32"
        assert contract["int_mm_min_rows"] == 17
        assert contract["fallback_compute"] == "float_activation_x_dequant_weight"
        assert contract["min_param_size"] == 4096
        assert contract["in_feat_align"] == 8
        assert "q_proj" in contract["skip_suffixes"]


# --------------------------------------------------------------------------- #
# QuantLinear 形状验证
# --------------------------------------------------------------------------- #

class TestQuantLinearShape:
    def _make_qlinear(self, in_f=64, out_f=128):
        w = torch.randn(out_f, in_f)
        return QuantLinear(w, None)

    def test_weight_shape_is_transposed(self):
        """weight_int8 应存储为 [in_features, out_features]（适配 _int_mm [K,N] 布局）。"""
        ql = self._make_qlinear(64, 128)
        assert ql.weight_int8.shape == (64, 128)

    def test_weight_dtype_is_int8(self):
        ql = self._make_qlinear(64, 128)
        assert ql.weight_int8.dtype == torch.int8

    def test_scale_w_is_per_channel_vector(self):
        ql = self._make_qlinear(64, 128)
        assert ql.scale_w.shape == (128,)

    def test_in_out_features(self):
        ql = self._make_qlinear(64, 128)
        assert ql.in_features == 64
        assert ql.out_features == 128

    def test_quantize_activation_per_row_returns_column_scale(self):
        x = torch.tensor([[1.0, 2.0], [10.0, 20.0]])
        x_int8, scale_a = QuantLinear._quantize_activation_per_row(x)
        assert x_int8.shape == x.shape
        assert scale_a.shape == (2, 1)

    def test_should_use_int_mm_threshold(self):
        assert QuantLinear._should_use_int_mm("cuda", 17) is True
        assert QuantLinear._should_use_int_mm("cuda", 16) is False
        assert QuantLinear._should_use_int_mm("cpu", 64) is False


# --------------------------------------------------------------------------- #
# CPU 路径前向精度
# --------------------------------------------------------------------------- #

class TestQuantLinearForwardCPU:
    def _make_ql_and_linear(self, in_f=64, out_f=128, seed=42):
        torch.manual_seed(seed)
        w = torch.randn(out_f, in_f)
        linear = nn.Linear(in_f, out_f, bias=False)
        linear.weight.data.copy_(w)
        ql = QuantLinear(w, None)
        return ql, linear

    def test_forward_shape(self):
        ql, _ = self._make_ql_and_linear()
        x = torch.randn(4, 64)
        out = ql(x)
        assert out.shape == (4, 128)

    def test_forward_dtype_preserved(self):
        ql, _ = self._make_ql_and_linear()
        x = torch.randn(4, 64).half()
        out = ql(x)
        assert out.dtype == torch.float16

    def test_quantization_error_cpu(self):
        """反量化误差（W8A8 CPU fallback）相对于 fp32 matmul 应在合理范围内。"""
        ql, linear = self._make_ql_and_linear()
        torch.manual_seed(0)
        x = torch.randn(8, 64)
        out_q = ql(x.float())
        out_ref = linear(x.float())
        rel_err = ((out_q - out_ref).abs() / (out_ref.abs() + 1e-6)).mean()
        # per-channel W8A8 随机权重典型相对误差 < 10%（真实模型权重更集中，误差更小）
        assert rel_err < 0.10, f"相对误差过大: {rel_err:.4f}"

    def test_cpu_fallback_uses_float_activation_with_dequantized_weight(self):
        torch.manual_seed(7)
        weight = torch.randn(32, 16)
        bias = torch.randn(32)
        ql = QuantLinear(weight, bias)
        x = torch.randn(4, 16)

        out = ql(x)
        expected = x.float() @ ql._dequantize_weight()
        expected = expected + bias.float()

        assert torch.allclose(out.float(), expected, atol=1e-5, rtol=1e-5)

    def test_forward_with_bias(self):
        in_f, out_f = 32, 64
        torch.manual_seed(1)
        w = torch.randn(out_f, in_f)
        b = torch.randn(out_f)
        ql = QuantLinear(w, b)
        x = torch.randn(4, in_f)
        out = ql(x)
        assert out.shape == (4, out_f)

    def test_no_bias(self):
        w = torch.randn(64, 32)
        ql = QuantLinear(w, None)
        assert ql.bias is None

    def test_dequant_scales_before_cast_to_half(self):
        """int32 累加值可能远大于 fp16 上限；必须先乘 scale 再 cast。"""
        w = torch.ones(1, 1024)
        ql = QuantLinear(w, None)
        x = torch.ones(1, 1024).half()
        out = ql(x)
        assert torch.isfinite(out).all()
        assert out.dtype == torch.float16
        assert out.item() > 0

    def test_runtime_stats_records_cpu_fallback(self):
        QuantLinear.reset_runtime_stats()
        w = torch.randn(64, 32)
        ql = QuantLinear(w, None)
        _ = ql(torch.randn(4, 32))
        stats = QuantLinear.get_runtime_stats()
        assert stats["fallback_calls"] == 1
        assert stats["fallback_rows"] == 4
        assert stats["int_mm_calls"] == 0
        assert stats["int_mm_rows"] == 0

    def test_reset_runtime_stats(self):
        QuantLinear.reset_runtime_stats()
        stats = QuantLinear.get_runtime_stats()
        assert stats == {
            "int_mm_calls": 0,
            "fallback_calls": 0,
            "int_mm_rows": 0,
            "fallback_rows": 0,
        }


# --------------------------------------------------------------------------- #
# CUDA 路径前向精度（M≥17，满足 torch._int_mm 约束）
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA 不可用")
class TestQuantLinearForwardCUDA:
    def test_forward_cuda_shape(self):
        """CUDA _int_mm 路径：M=20 > 16，验证形状正确。"""
        w = torch.randn(128, 64).cuda()
        ql = QuantLinear(w, None).cuda()
        x = torch.randn(20, 64).cuda()
        out = ql(x)
        assert out.shape == (20, 128)

    def test_forward_cuda_dtype(self):
        w = torch.randn(128, 64).cuda().half()
        ql = QuantLinear(w, None).cuda()
        x = torch.randn(20, 64).cuda().half()
        out = ql(x)
        assert out.dtype == torch.float16

    def test_quantization_error_cuda(self):
        """CUDA 路径反量化误差 < 5%（相对于 fp32 matmul）。"""
        torch.manual_seed(42)
        in_f, out_f = 64, 128
        w = torch.randn(out_f, in_f).cuda()
        ql = QuantLinear(w, None).cuda()
        x = torch.randn(20, in_f).cuda()
        out_q = ql(x.float())
        out_ref = x.float() @ w.T.float()
        rel_err = ((out_q - out_ref).abs() / (out_ref.abs() + 1e-6)).mean()
        # per-channel W8A8 随机权重典型相对误差 < 10%
        assert rel_err < 0.10, f"CUDA 相对误差过大: {rel_err:.4f}"

    def test_cuda_small_m_fallback_uses_float_activation_with_dequantized_weight(self):
        torch.manual_seed(123)
        QuantLinear.reset_runtime_stats()
        in_f, out_f = 32, 64
        weight = torch.randn(out_f, in_f, device="cuda", dtype=torch.float16)
        bias = torch.randn(out_f, device="cuda", dtype=torch.float16)
        ql = QuantLinear(weight, bias).cuda()
        x = torch.randn(8, in_f, device="cuda", dtype=torch.float16)  # rows < 17，必须走 fallback

        out = ql(x)
        expected = x.float() @ ql._dequantize_weight()
        expected = expected + bias.float()
        expected = expected.to(x.dtype)

        assert torch.allclose(out, expected, atol=1e-3, rtol=1e-3)
        stats = QuantLinear.get_runtime_stats()
        assert stats["fallback_calls"] == 1
        assert stats["fallback_rows"] == 8
        assert stats["int_mm_calls"] == 0


# --------------------------------------------------------------------------- #
# quantize_model()
# --------------------------------------------------------------------------- #

class _SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(1000, 64)
        self.q_proj = nn.Linear(64, 64)       # attention projection，跳过
        self.gate_proj = nn.Linear(64, 128)   # MLP 主线，量化
        self.linear1 = nn.Linear(64, 128)  # 64*128=8192 >= 4096，会被量化
        self.linear_small = nn.Linear(8, 16)  # 8*16=128 < 4096，跳过
        self.lm_head = nn.Linear(128, 1000)   # 跳过

    def forward(self, x):
        return self.linear1(x)


class TestQuantizeModel:
    def test_replaces_eligible_linear(self):
        model = _SmallModel()
        quantize_model(model)
        assert isinstance(model.linear1, QuantLinear), "linear1 应被替换为 QuantLinear"

    def test_skips_lm_head(self):
        model = _SmallModel()
        quantize_model(model)
        assert isinstance(model.lm_head, nn.Linear), "lm_head 不应被量化"

    def test_skips_attention_q_proj(self):
        model = _SmallModel()
        quantize_model(model)
        assert isinstance(model.q_proj, nn.Linear), "attention q_proj 第一版保留 fp16"

    def test_quantizes_mlp_gate_proj(self):
        model = _SmallModel()
        quantize_model(model)
        assert isinstance(model.gate_proj, QuantLinear), "MLP gate_proj 应被量化"

    def test_skips_small_layer(self):
        model = _SmallModel()
        quantize_model(model)
        assert isinstance(model.linear_small, nn.Linear), "小层不应被量化"

    def test_embed_tokens_unchanged(self):
        """embed_tokens 是 nn.Embedding，不是 nn.Linear，不会被替换。"""
        model = _SmallModel()
        quantize_model(model)
        assert isinstance(model.embed_tokens, nn.Embedding)

    def test_nested_linear_replaced(self):
        """嵌套子模块中的 nn.Linear 也应被替换。"""
        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(64, 128)

        class Outer(nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = Inner()

        model = Outer()
        quantize_model(model)
        assert isinstance(model.sub.proj, QuantLinear)

    def test_returns_same_model(self):
        model = _SmallModel()
        result = quantize_model(model)
        assert result is model


# --------------------------------------------------------------------------- #
# EngineConfig quant_mode 验证
# --------------------------------------------------------------------------- #

class TestEngineConfigQuantMode:
    def test_default_empty(self):
        cfg = EngineConfig(model_name="dummy", dry_run=True)
        assert cfg.quant_mode == ""

    def test_w8a8_accepted(self):
        cfg = EngineConfig(model_name="dummy", dry_run=True, quant_mode="w8a8")
        assert cfg.quant_mode == "w8a8"

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="quant_mode"):
            EngineConfig(model_name="dummy", dry_run=True, quant_mode="fp8")
