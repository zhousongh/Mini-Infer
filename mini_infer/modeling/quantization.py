"""Phase 16：W8A8 / mixed-fallback 量化推理核心路径。

提供三个公共接口：
- QuantMode：量化模式枚举（""  / "w8a8"）
- QuantLinear：替换 nn.Linear 的 W8A8 量化线性层
- quantize_model()：递归替换模型中的 nn.Linear → QuantLinear

当前范围：
- CUDA 大 M（rows >= 17）：真正的 W8A8（activation per-row + weight per-channel）+ PyTorch 原生 _int_mm
- CUDA 小 M / CPU：保留 int8 权重存储，但 compute 走高保真 fallback
  （float activation x dequantized weight）

因此当前 Phase 16 的 engine benchmark 在 fallback 主导时，衡量的是
int8 权重存储 + mixed compute path，而不是纯 A8 matmul。
FP8 与 Triton INT8 GEMM 作为后续扩展，不计入本阶段。
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar, Optional

import torch
import torch.nn as nn


class QuantMode(str, Enum):
    """量化模式。空字符串表示不量化，保持 fp16 原路径。"""
    NONE = ""
    W8A8 = "w8a8"

    @classmethod
    def from_str(cls, mode: str) -> "QuantMode":
        for m in cls:
            if m.value == mode:
                return m
        raise ValueError(f"不支持的 quant_mode: {mode!r}，合法值: {[m.value for m in cls]}")


class QuantLinear(nn.Module):
    """W8A8 量化线性层，替换 nn.Linear。

    权重量化：按输出通道一次性完成，scale_w[i] = max(|W[i]|) / 127。
    计算路径：
        GPU 大 M: activation per-row -> torch._int_mm(x_int8 [M,K], weight_int8 [K,N])
                  -> int32 -> fp32 dequant -> 原始 dtype
        GPU 小 M / CPU: x_fp32 @ dequant(weight_int8, scale_w)（高保真 fallback）
    """

    _runtime_stats: ClassVar[dict[str, int]] = {
        "int_mm_calls": 0,
        "fallback_calls": 0,
        "int_mm_rows": 0,
        "fallback_rows": 0,
    }
    _contract: ClassVar[dict[str, object]] = {
        "weight_storage": "int8_per_channel",
        "int_mm_activation_granularity": "per_row",
        "fallback_activation_granularity": "fp32",
        "int_mm_min_rows": 17,
        "fallback_compute": "float_activation_x_dequant_weight",
        "skip_suffixes": ("lm_head", "embed_tokens", "q_proj", "k_proj", "v_proj", "o_proj"),
        "min_param_size": 4096,
        "in_feat_align": 8,
    }

    def __init__(self, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> None:
        """用 nn.Linear 的 weight/bias 初始化。

        Args:
            weight: 原始 fp16/fp32 权重，shape [out_features, in_features]（nn.Linear 标准）
            bias: 可选 bias，shape [out_features]
        """
        super().__init__()
        out_features, in_features = weight.shape

        weight_int8_T, scale_w = self._quantize_weight_per_channel(weight)

        # 注册为 buffer（不参与梯度、随模型 state_dict 保存）
        self.register_buffer("weight_int8", weight_int8_T)  # [K, N]
        self.register_buffer("scale_w", scale_w)

        if bias is not None:
            self.register_buffer("bias", bias.clone())
        else:
            self.bias = None  # type: ignore[assignment]

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """W8A8 量化前向，输出 dtype 与输入一致。

        支持 2D [M, K] 和 3D [B, S, K] 输入（自动 reshape）。
        """
        orig_dtype = x.dtype
        orig_shape = x.shape
        # 展平到 2D：[*, K] → [M, K]
        x_2d = x.reshape(-1, orig_shape[-1])
        x_fp = x_2d.float()

        # 矩阵乘：CPU fallback 或 CUDA _int_mm
        M = x_fp.shape[0]
        if not self._should_use_int_mm(x.device.type, M):
            # CPU 不支持 _int_mm；CUDA 小 M（如 decode）也走此路径。
            # 这里不再量化激活，而是直接用 fp32 激活乘反量化后的权重：
            #   1. 当前 engine workload 中 prefill/decode 基本都是 fallback，优先保证正确性
            #   2. 仍保留 int8 权重存储，因此显存收益不变
            self.__class__._runtime_stats["fallback_calls"] += 1
            self.__class__._runtime_stats["fallback_rows"] += int(M)
            out = x_fp @ self._dequantize_weight().float()  # [M, N] float32
        else:
            # CUDA 路径：_int_mm INT8 GEMM，适合 prefill / 大 batch decode
            x_int8, scale_a = self._quantize_activation_per_row(x_fp)
            self.__class__._runtime_stats["int_mm_calls"] += 1
            self.__class__._runtime_stats["int_mm_rows"] += int(M)
            out = torch._int_mm(x_int8.contiguous(), self.weight_int8)  # [M, N] int32

            # 反量化：先在 fp32 中乘 scale，避免 int32 累加结果直接 cast 到 fp16/bf16 溢出
            dequant_scale = scale_a.float() * self.scale_w.float()
            out = out.float() * dequant_scale.float()

        if self.bias is not None:
            out = out + self.bias.float()

        out = out.to(orig_dtype)

        # 恢复原始形状（如 3D 输入）
        if len(orig_shape) > 2:
            out = out.reshape(*orig_shape[:-1], self.out_features)

        return out

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, W8A8"

    @classmethod
    def reset_runtime_stats(cls) -> None:
        """重置运行时统计，供 benchmark 区分 prefill / decode 路径使用。"""
        for key in cls._runtime_stats:
            cls._runtime_stats[key] = 0

    @classmethod
    def get_runtime_stats(cls) -> dict[str, int]:
        """返回当前运行时统计快照。"""
        return dict(cls._runtime_stats)

    @classmethod
    def get_contract(cls) -> dict[str, object]:
        """返回第一版 W8A8 contract，供测试和 benchmark 解释使用。"""
        return dict(cls._contract)

    @classmethod
    def _quantize_weight_per_channel(
        cls,
        weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """按输出通道量化权重，返回 [K, N] int8 权重和 [N] scale。"""
        abs_max = weight.abs().amax(dim=1).float()  # [out_features]
        scale_w = torch.clamp(abs_max / 127.0, min=1e-8)
        weight_fp = weight.float()
        weight_int8 = (
            weight_fp / scale_w.unsqueeze(1)
        ).round().clamp(-128, 127).to(torch.int8)
        return weight_int8.T.contiguous(), scale_w

    def _dequantize_weight(self) -> torch.Tensor:
        """把 [K, N] int8 权重恢复成 float32，供高保真 fallback 使用。"""
        return self.weight_int8.float() * self.scale_w.float().unsqueeze(0)

    @classmethod
    def _quantize_activation_per_row(
        cls,
        x_fp: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """按输入行量化激活，返回 int8 激活和 [M,1] scale。"""
        abs_max_a = x_fp.abs().amax(dim=1, keepdim=True)
        scale_a = torch.clamp(abs_max_a / 127.0, min=1e-8)
        x_int8 = (x_fp / scale_a).round().clamp(-128, 127).to(torch.int8)
        return x_int8, scale_a

    @classmethod
    def _should_use_int_mm(cls, device_type: str, rows: int) -> bool:
        """判断本次前向是否应命中 CUDA _int_mm 快路径。"""
        return device_type != "cpu" and rows >= int(cls._contract["int_mm_min_rows"])


# --------------------------------------------------------------------------- #
# 跳过量化的条件
# --------------------------------------------------------------------------- #

# 按层名后缀跳过：embed/output head 保持 fp16
# Attention 投影层（q/k/v/o_proj）对第一版 W8A8 更敏感，跳过保留 fp16 以保证正确性
# MLP 层（gate_proj / up_proj / down_proj）权重较大且分布较均匀，是量化主要收益来源
_SKIP_SUFFIXES = tuple(QuantLinear.get_contract()["skip_suffixes"])

# 参数量下限（太小的层 INT8 GEMM 收益极低，且可能触发 _int_mm 的 size 约束）
_MIN_PARAM_SIZE = int(QuantLinear.get_contract()["min_param_size"])

# in_features 必须是 8 的倍数（INT8 GEMM 对齐要求）
_IN_FEAT_ALIGN = int(QuantLinear.get_contract()["in_feat_align"])


def _should_skip(name: str, module: nn.Linear) -> bool:
    """判断是否跳过量化。"""
    for suffix in _SKIP_SUFFIXES:
        if name == suffix or name.endswith(f".{suffix}"):
            return True
    in_f, out_f = module.in_features, module.out_features
    if in_f * out_f < _MIN_PARAM_SIZE:
        return True
    if in_f % _IN_FEAT_ALIGN != 0:
        return True
    return False


def quantize_model(model: nn.Module) -> nn.Module:
    """原地替换模型中的 nn.Linear → QuantLinear（W8A8 MLP-only）。

    跳过规则（来自 QuantLinear._contract["skip_suffixes"]）：
    - 名称后缀为 lm_head / embed_tokens（输入/输出层保持 fp16）
    - 名称后缀为 q_proj / k_proj / v_proj / o_proj（attention 投影层对
      per-channel W8A8 更敏感，保守保留 fp16）
    - in_features × out_features < min_param_size（当前 4096，太小的层无收益）
    - in_features % in_feat_align != 0（当前 8，INT8 GEMM 对齐要求）

    实际被量化的层：MLP 主线 gate_proj / up_proj / down_proj（以及其他非跳过的大型线性层）

    Args:
        model: 已加载到目标 device 的 fp16/fp32 模型（原地修改）

    Returns:
        修改后的同一 model 对象
    """
    _replace_linear(model, "")
    return model


def _replace_linear(module: nn.Module, prefix: str) -> None:
    """DFS 递归替换，prefix 为点分层名（不含末尾点）。"""
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            if not _should_skip(full_name, child):
                qlinear = QuantLinear(child.weight.data, child.bias.data if child.bias is not None else None)
                setattr(module, name, qlinear)
        else:
            _replace_linear(child, full_name)
