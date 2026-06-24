"""mini_infer.modeling — 模型执行器、量化、MLA、MoE。"""

from .mla_attention import (
    MLAAttentionAbsorbed,
    MLAAttentionLatentCache,
    MLAAttentionNaive,
)
from .model_runner import ModelRunner
from .quantization import QuantLinear, QuantMode, quantize_model

__all__ = [
    "ModelRunner",
    "QuantLinear", "QuantMode", "quantize_model",
    "MLAAttentionNaive", "MLAAttentionLatentCache", "MLAAttentionAbsorbed",
]

try:
    from .moe_layer import EPMoELayer, MoELayer, shard_moe_state_dict
    from .moe_model import SyntheticMoEConfig, SyntheticMoEModel
except ModuleNotFoundError as exc:
    if exc.name not in {
        f"{__name__}.moe_layer",
        f"{__name__}.moe_model",
    }:
        raise
else:
    __all__ += [
        "MoELayer", "EPMoELayer", "shard_moe_state_dict",
        "SyntheticMoEConfig", "SyntheticMoEModel",
    ]
