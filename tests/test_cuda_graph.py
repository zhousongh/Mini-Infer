"""
Phase 12 CUDA Graph 单元测试（dry_run 路径）。

覆盖范围：
  - warmup_cuda_graphs 在 dry_run=True 时不改变 graph pool
  - _find_padded_bs：空 pool 返回 None；正确找到最小 padded bs；超出范围返回 None
  - decode_batch 在空 graph pool 时仍走 eager（dry_run）路径
  - use_cuda_graph=True 在 dry_run 引擎中不触发 GPU 操作
"""

import pytest

from mini_infer.core.config import EngineConfig
from mini_infer.runtime.engine import LLMEngine
from mini_infer.cache.kv_cache import KVCacheManager
from mini_infer.modeling.model_runner import ModelRunner


def _dry_runner() -> ModelRunner:
    cfg = EngineConfig(model_name="dummy", dry_run=True)
    kv = KVCacheManager(config=cfg)
    return ModelRunner(config=cfg, kv_cache=kv)


# ------------------------------------------------------------------
# warmup_cuda_graphs dry_run 路径
# ------------------------------------------------------------------

def test_warmup_dry_run_is_noop():
    """dry_run=True 时 warmup_cuda_graphs 不填充 graph pool。"""
    runner = _dry_runner()
    assert runner._cuda_graphs == {}
    runner.warmup_cuda_graphs()
    assert runner._cuda_graphs == {}


# ------------------------------------------------------------------
# _find_padded_bs
# ------------------------------------------------------------------

def test_find_padded_bs_empty_pool():
    """空 pool 时返回 None（走 eager 路径）。"""
    runner = _dry_runner()
    assert runner._find_padded_bs(1) is None
    assert runner._find_padded_bs(8) is None


def test_find_padded_bs_exact_match():
    """batch_size 恰好匹配捕获的 bs。"""
    runner = _dry_runner()
    runner._cuda_graphs = {1: object(), 4: object(), 8: object()}
    assert runner._find_padded_bs(1) == 1
    assert runner._find_padded_bs(4) == 4
    assert runner._find_padded_bs(8) == 8


def test_find_padded_bs_needs_padding():
    """batch_size 不完全匹配时，返回最小的 >= actual_bs 的 padded bs。"""
    runner = _dry_runner()
    runner._cuda_graphs = {1: object(), 4: object(), 8: object()}
    assert runner._find_padded_bs(2) == 4
    assert runner._find_padded_bs(3) == 4
    assert runner._find_padded_bs(5) == 8
    assert runner._find_padded_bs(7) == 8


def test_find_padded_bs_exceeds_pool():
    """batch_size 超出 pool 上限时返回 None。"""
    runner = _dry_runner()
    runner._cuda_graphs = {1: object(), 4: object()}
    assert runner._find_padded_bs(5) is None
    assert runner._find_padded_bs(8) is None


# ------------------------------------------------------------------
# decode_batch 在 dry_run + 空 graph pool 时正常运行
# ------------------------------------------------------------------

def test_decode_batch_empty_graph_pool_dry_run():
    """空 graph pool 的 dry_run decode 正常推进请求状态。"""
    cfg = EngineConfig(model_name="dummy", dry_run=True, use_cuda_graph=True)
    engine = LLMEngine(cfg)
    outputs = engine.generate(["hello"], max_new_tokens=3)
    assert len(outputs) == 1
    assert len(outputs[0]) > 0


# ------------------------------------------------------------------
# use_cuda_graph=True + dry_run=True 引擎初始化不报错
# ------------------------------------------------------------------

def test_engine_init_cuda_graph_dry_run():
    """use_cuda_graph=True 在 dry_run 引擎中初始化不触发 GPU 操作，不报错。"""
    cfg = EngineConfig(model_name="dummy", dry_run=True, use_cuda_graph=True)
    engine = LLMEngine(cfg)
    # graph pool 应该为空（dry_run 下 warmup 是 noop）
    assert engine.model_runner._cuda_graphs == {}
