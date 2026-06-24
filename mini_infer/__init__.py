"""这个文件导出 mini-infer 的公开接口，供外部统一导入。"""

from __future__ import annotations

from importlib import import_module

from .core.config import EngineConfig
from .core.request import Request, RequestState, SamplingParams

_LAZY_IMPORTS = {
    "LLMEngine":      (".runtime.engine",       "LLMEngine"),
    "AsyncEngine":    (".runtime.async_engine",  "AsyncEngine"),
    "SpecEngine":     (".runtime.spec_engine",   "SpecEngine"),
    "PDEngine":       (".runtime.pd_engine",     "PDEngine"),
    "PPEngine":       (".parallel.pp_engine",    "PPEngine"),
    "ReplicaEngine":  (".parallel.replica_engine", "ReplicaEngine"),
    "TPEngine":       (".parallel.tp_engine",    "TPEngine"),
    "EPEngine":       (".parallel.ep_engine",    "EPEngine"),
    "QuantLinear":    (".modeling.quantization", "QuantLinear"),
    "quantize_model": (".modeling.quantization", "quantize_model"),
    "QuantMode":      (".modeling.quantization", "QuantMode"),
}

__all__ = [
    "EngineConfig",
    "LLMEngine",
    "AsyncEngine",       # Phase 8：异步 HTTP 服务引擎
    "PPEngine",          # Phase 4：HF Pipeline Parallel（测量用）
    "ReplicaEngine",     # Phase 4：数据并行副本
    "SpecEngine",        # Phase 11：Speculative Decoding
    "TPEngine",          # Phase 13：Tensor Parallelism（真 TP，NCCL all-reduce）
    "PDEngine",          # Phase 15：Disaggregated Prefill/Decode
    "EPEngine",          # Phase 17：Expert Parallelism（2 卡 all-to-all 原型）
    "QuantLinear",       # Phase 16：W8A8 量化线性层
    "quantize_model",    # Phase 16：模型量化入口
    "QuantMode",         # Phase 16：量化模式枚举
    "Request",
    "RequestState",
    "SamplingParams",
]

__version__ = "0.21.0"


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
