"""Phase 16 ModelRunner 量化接入测试。

覆盖范围：
  - quant_mode="" 时不调用 quantize_model
  - quant_mode="w8a8" 时先 quantize_model，再 patch paged attention
"""

from __future__ import annotations

import torch.nn as nn
import transformers

import mini_infer.kernels.attention as attention_mod
import mini_infer.modeling.quantization as quant_mod
from mini_infer.core.config import EngineConfig
from mini_infer.cache.kv_cache import KVCacheManager
from mini_infer.modeling.model_runner import ModelRunner


class _FakeTokenizer:
    pad_token_id = None
    eos_token_id = 1
    eos_token = "<eos>"
    pad_token = None


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quantized = False

    def eval(self):
        return self


def _tiny_cfg(quant_mode: str) -> EngineConfig:
    return EngineConfig(
        model_name="dummy",
        device="cpu",
        dtype="float32",
        dry_run=False,
        quant_mode=quant_mode,
        block_size=1,
        num_gpu_blocks=1,
        num_hidden_layers=1,
        num_kv_heads=1,
        head_dim=1,
    )


def test_model_runner_skips_quantize_model_when_quant_mode_empty(monkeypatch):
    events: list[object] = []

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _FakeTokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: _FakeModel(),
    )

    def fake_quantize_model(model):
        events.append("quantize")
        model.quantized = True
        return model

    def fake_patch_model_for_paged_decode(model, kv_cache):
        events.append(("patch", model.quantized))
        return object()

    monkeypatch.setattr(quant_mod, "quantize_model", fake_quantize_model)
    monkeypatch.setattr(attention_mod, "patch_model_for_paged_decode", fake_patch_model_for_paged_decode)

    cfg = _tiny_cfg(quant_mode="")
    kv = KVCacheManager(cfg)
    _ = ModelRunner(cfg, kv)

    assert events == [("patch", False)]


def test_model_runner_quantizes_before_paged_attention_patch(monkeypatch):
    events: list[object] = []

    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _FakeTokenizer(),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM,
        "from_pretrained",
        lambda *args, **kwargs: _FakeModel(),
    )

    def fake_quantize_model(model):
        events.append("quantize")
        model.quantized = True
        return model

    def fake_patch_model_for_paged_decode(model, kv_cache):
        events.append(("patch", model.quantized))
        return object()

    monkeypatch.setattr(quant_mod, "quantize_model", fake_quantize_model)
    monkeypatch.setattr(attention_mod, "patch_model_for_paged_decode", fake_patch_model_for_paged_decode)

    cfg = _tiny_cfg(quant_mode="w8a8")
    kv = KVCacheManager(cfg)
    _ = ModelRunner(cfg, kv)

    assert events == ["quantize", ("patch", True)]
