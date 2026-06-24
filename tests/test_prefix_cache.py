"""
Phase 10 Prefix Cache 测试。

覆盖范围：
  - find_prefix_cache：miss / partial hit / full hit（capped）
  - init_request_with_prefix：引用计数、block_table 结构
  - free_request：ref_count 控制释放（共享块不提前归还）
  - register_prefix_blocks_for_request：新块写入缓存 + LRU 更新
  - evict_lru_prefix_block：仅淘汰 ref_count == 1 的 block
  - LRU eviction 在 _allocate_block 压力下触发
  - engine.generate 端到端：prefix miss → hit 路径均可完成生成，prefix_cache_size 可读
"""

from __future__ import annotations

from mini_infer.core.config import EngineConfig
from mini_infer.runtime.engine import LLMEngine
from mini_infer.cache.kv_cache import KVCacheManager
from mini_infer.core.request import Request, RequestState, SamplingParams


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _make_manager(num_blocks: int = 10, block_size: int = 4) -> KVCacheManager:
    config = EngineConfig(
        model_name="stub",
        dry_run=True,
        num_gpu_blocks=num_blocks,
        block_size=block_size,
    )
    return KVCacheManager(config=config)


def _make_state(token_ids: list[int], request_id: str = "r1") -> RequestState:
    req = Request(
        request_id=request_id,
        prompt="",
        sampling_params=SamplingParams(max_new_tokens=8),
    )
    state = RequestState(request=req, prompt_token_ids=token_ids)
    return state


def _make_engine(num_gpu_blocks: int = 32, block_size: int = 4) -> LLMEngine:
    config = EngineConfig(
        model_name="stub",
        dry_run=True,
        num_gpu_blocks=num_gpu_blocks,
        block_size=block_size,
    )
    return LLMEngine(config)


# ---------------------------------------------------------------------------
# find_prefix_cache：基础 miss/hit
# ---------------------------------------------------------------------------

class TestFindPrefixCache:
    def test_miss_on_empty_cache(self) -> None:
        mgr = _make_manager()
        cached_len, cached_blocks = mgr.find_prefix_cache([1, 2, 3, 4])
        assert cached_len == 0
        assert cached_blocks == []

    def test_partial_hit(self) -> None:
        """注册 5 个 token 的 prompt（block_size=4，第一个 block=[1,2,3,4]）后，
        8 token prompt 中前 4 个命中 1 个 block。

        注意：block_size=4 时，恰好 4 token 的 prompt 因 capping 不写入缓存（无法保证非空 suffix）。
        使用 5-token 初始 prompt 可注册第一个完整 block。
        """
        mgr = _make_manager(block_size=4)
        token_ids_5 = [1, 2, 3, 4, 5]  # 5 tokens → 1 full block cached
        state = _make_state(token_ids_5, "r1")
        mgr.init_request(state)
        mgr.register_prefix_blocks(token_ids_5, mgr._block_tables["r1"])
        mgr.free_request(state)

        # 8-token prompt 包含相同前 4 个 token，应命中 1 block
        token_ids_8 = [1, 2, 3, 4, 5, 6, 7, 8]
        cached_len, cached_blocks = mgr.find_prefix_cache(token_ids_8)
        assert cached_len == 4
        assert len(cached_blocks) == 1

    def test_full_prompt_capped_to_avoid_empty_suffix(self) -> None:
        """block 对齐的 prompt 最多缓存 num_full_blocks - 1 个 block，保证非空 suffix。"""
        mgr = _make_manager(block_size=4)
        token_ids = [1, 2, 3, 4, 5, 6, 7, 8]  # 正好 2 个 block
        state = _make_state(token_ids, "r1")
        mgr.init_request(state)
        mgr.register_prefix_blocks(token_ids, mgr._block_tables["r1"])
        mgr.free_request(state)

        # find_prefix_cache 对同样 8 token 只应命中 1 个 block（capped）
        cached_len, cached_blocks = mgr.find_prefix_cache(token_ids)
        assert cached_len == 4       # 只命中前 1 block，不是全部 2 block
        assert len(cached_blocks) == 1


# ---------------------------------------------------------------------------
# init_request_with_prefix + ref_count
# ---------------------------------------------------------------------------

class TestInitRequestWithPrefix:
    def test_block_table_contains_cached_plus_new_blocks(self) -> None:
        """prefix hit 请求的 block_table = cached_blocks + new suffix blocks。"""
        mgr = _make_manager(num_blocks=10, block_size=4)
        # 先建立 prefix cache（5-token prompt 注册第一个 block [1,2,3,4]）
        token_ids_5 = [1, 2, 3, 4, 5]
        state1 = _make_state(token_ids_5, "r1")
        mgr.init_request(state1)
        mgr.register_prefix_blocks(token_ids_5, mgr._block_tables["r1"])
        mgr.free_request(state1)

        # 8-token 请求，前 4 命中
        token_ids_8 = [1, 2, 3, 4, 5, 6, 7, 8]
        cached_len, cached_blocks = mgr.find_prefix_cache(token_ids_8)
        assert cached_len == 4

        state2 = _make_state(token_ids_8, "r2")
        mgr.init_request_with_prefix(state2, cached_len, cached_blocks)

        bt = mgr._block_tables["r2"]
        # 前 cached_len // block_size 个块来自 cache
        assert bt[:1] == cached_blocks
        # 后面有新分配的块用于 suffix
        assert len(bt) >= 2

    def test_ref_count_incremented_for_cached_blocks(self) -> None:
        """prefix hit 时被复用的 cached block 引用计数必须 +1。"""
        mgr = _make_manager(num_blocks=10, block_size=4)
        token_ids_5 = [1, 2, 3, 4, 5]  # 5 tokens → registers block [1,2,3,4]
        state1 = _make_state(token_ids_5, "r1")
        mgr.init_request(state1)
        mgr.register_prefix_blocks(token_ids_5, mgr._block_tables["r1"])
        mgr.free_request(state1)

        query_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        cached_len, cached_blocks = mgr.find_prefix_cache(query_ids)
        assert cached_len == 4
        phys_block = cached_blocks[0]
        ref_before = mgr._ref_count.get(phys_block, 0)

        state2 = _make_state(query_ids, "r2")
        mgr.init_request_with_prefix(state2, cached_len, cached_blocks)

        ref_after = mgr._ref_count.get(phys_block, 0)
        assert ref_after == ref_before + 1

    def test_shared_block_not_freed_until_last_reference(self) -> None:
        """cached block 被两个请求共享时，释放第一个请求不应把块还给 free pool。"""
        mgr = _make_manager(num_blocks=10, block_size=4)
        token_ids_5 = [1, 2, 3, 4, 5]  # 5 tokens → registers block [1,2,3,4]
        state1 = _make_state(token_ids_5, "r1")
        mgr.init_request(state1)
        mgr.register_prefix_blocks(token_ids_5, mgr._block_tables["r1"])
        mgr.free_request(state1)

        # r2 和 r3 都命中相同前缀
        token_ids_8 = [1, 2, 3, 4, 5, 6, 7, 8]
        cached_len, cached_blocks = mgr.find_prefix_cache(token_ids_8)
        assert cached_len == 4
        phys_block = cached_blocks[0]

        state2 = _make_state(token_ids_8, "r2")
        mgr.init_request_with_prefix(state2, cached_len, cached_blocks)

        state3 = _make_state(token_ids_8, "r3")
        mgr.init_request_with_prefix(state3, cached_len, list(cached_blocks))

        mgr.free_request(state2)
        # shared block 仍被 cache 和 r3 持有，不应出现在 free pool
        assert phys_block not in list(mgr._free_blocks)


# ---------------------------------------------------------------------------
# evict_lru_prefix_block
# ---------------------------------------------------------------------------

class TestEvictLRU:
    def test_evict_oldest_when_free_pool_exhausted(self) -> None:
        """空闲块耗尽时，_allocate_block 自动触发 LRU eviction。

        num_blocks=4, block_size=4。两条 5-token prompt 各注册 1 个 block（共 2 个块），
        用完所有空闲块后分配第三条请求时触发 eviction。
        """
        mgr = _make_manager(num_blocks=4, block_size=4)

        # 建立两条 prefix cache 记录（各 5 token → 注册 1 block）
        for rid, tids in [("r1", [1, 2, 3, 4, 5]), ("r2", [5, 6, 7, 8, 9])]:
            state = _make_state(tids, rid)
            mgr.init_request(state)
            mgr.register_prefix_blocks(tids, mgr._block_tables[rid])
            mgr.free_request(state)

        # 两条 prompt 各用 2 块（1 block-aligned + 1 suffix）；release 后 cache 持有各 1 块
        # 实际空闲块数取决于分配情况，prefix_cache_size 至少 >= 1
        assert mgr.prefix_cache_size() >= 1

        # 耗尽全部空闲块
        free_before = mgr.num_free_blocks()
        filler_states = []
        for i in range(free_before):
            s = _make_state([100 + i], f"filler{i}")
            mgr.init_request(s)
            filler_states.append(s)

        # 再分配一个新请求，应触发 LRU eviction
        cache_before = mgr.prefix_cache_size()
        state_new = _make_state([200, 201, 202, 203, 204], "r_new")
        mgr.init_request(state_new)  # 触发 evict
        assert mgr.prefix_cache_size() < cache_before

    def test_evict_only_ref_count_one_blocks(self) -> None:
        """ref_count > 1 的 block（被活跃请求使用）不得被 evict。"""
        mgr = _make_manager(num_blocks=4, block_size=4)
        # 5-token prompt → registers block [1,2,3,4]
        token_ids_5 = [1, 2, 3, 4, 5]
        state1 = _make_state(token_ids_5, "r1")
        mgr.init_request(state1)
        mgr.register_prefix_blocks(token_ids_5, mgr._block_tables["r1"])
        mgr.free_request(state1)

        # r2 复用 r1 的 prefix（ref_count 变为 cache_ref + r2_ref = 2）
        token_ids_8 = [1, 2, 3, 4, 5, 6, 7, 8]
        cached_len, cached_blocks = mgr.find_prefix_cache(token_ids_8)
        assert cached_len == 4
        phys_block = cached_blocks[0]

        state2 = _make_state(token_ids_8, "r2")
        mgr.init_request_with_prefix(state2, cached_len, cached_blocks)

        # ref_count should be 2 (cache + r2)
        assert mgr._ref_count.get(phys_block, 0) >= 2

        # evict 应跳过这个 block（唯一 cached block 被 r2 引用，不可淘汰）
        result = mgr.evict_lru_prefix_block()
        assert result is False
        assert phys_block in mgr._pfx._cache.values()


# ---------------------------------------------------------------------------
# prefix-hit + preemption（un-admit）路径
# ---------------------------------------------------------------------------

class TestPrefixHitPreemption:
    def test_unadmit_prefix_hit_request_correct_refcount(self) -> None:
        """prefix-hit 请求被 un-admit 后，cached block 的引用计数正确恢复到 1（仅 cache 持有）。"""
        mgr = _make_manager(num_blocks=10, block_size=4)
        # 建立 prefix cache
        token_ids_5 = [1, 2, 3, 4, 5]
        state1 = _make_state(token_ids_5, "r1")
        mgr.init_request(state1)
        mgr.register_prefix_blocks(token_ids_5, mgr._block_tables["r1"])
        mgr.free_request(state1)

        # r2 命中 prefix
        token_ids_8 = [1, 2, 3, 4, 5, 6, 7, 8]
        cached_len, cached_blocks = mgr.find_prefix_cache(token_ids_8)
        assert cached_len == 4
        phys_block = cached_blocks[0]

        state2 = _make_state(token_ids_8, "r2")
        mgr.init_request_with_prefix(state2, cached_len, cached_blocks)
        state2.prefix_cached_len = cached_len
        state2.prefix_cached_blocks = list(cached_blocks)

        # ref_count should be 2 (cache + r2)
        assert mgr._ref_count.get(phys_block, 0) == 2

        # 模拟 un-admit：free_request
        mgr.free_request(state2)

        # 释放后 ref_count 降回 1（cache 仍持有）
        assert mgr._ref_count.get(phys_block, 0) == 1
        assert phys_block not in list(mgr._free_blocks)  # 不应还回 free pool

    def test_readmit_after_unadmit_uses_prefix_again(self) -> None:
        """un-admit 再重新准入后，prefix cache 仍可命中，状态一致。"""
        mgr = _make_manager(num_blocks=10, block_size=4)
        token_ids_5 = [1, 2, 3, 4, 5]
        state1 = _make_state(token_ids_5, "r1")
        mgr.init_request(state1)
        mgr.register_prefix_blocks(token_ids_5, mgr._block_tables["r1"])
        mgr.free_request(state1)

        token_ids_8 = [1, 2, 3, 4, 5, 6, 7, 8]
        # 第一次准入
        cached_len, cached_blocks = mgr.find_prefix_cache(token_ids_8)
        state2 = _make_state(token_ids_8, "r2")
        mgr.init_request_with_prefix(state2, cached_len, cached_blocks)
        state2.prefix_cached_len = cached_len
        state2.prefix_cached_blocks = list(cached_blocks)

        # un-admit（preemption not prefilled）
        mgr.free_request(state2)
        state2.prefix_cached_len = 0
        state2.prefix_cached_blocks = []

        # 重新准入：prefix cache 仍可命中
        cached_len2, cached_blocks2 = mgr.find_prefix_cache(token_ids_8)
        assert cached_len2 == 4  # 仍命中
        mgr.init_request_with_prefix(state2, cached_len2, cached_blocks2)
        state2.prefix_cached_len = cached_len2
        state2.prefix_cached_blocks = list(cached_blocks2)

        # 重新准入后 ref_count 再次为 2
        assert mgr._ref_count.get(cached_blocks2[0], 0) == 2

    def test_swap_out_clears_prefix_state(self) -> None:
        """swap_out 后 state.prefix_cached_len/blocks 被清空，避免指向已归还的物理块。"""
        mgr = _make_manager(num_blocks=10, block_size=4)
        token_ids_5 = [1, 2, 3, 4, 5]
        state1 = _make_state(token_ids_5, "r1")
        mgr.init_request(state1)
        mgr.register_prefix_blocks(token_ids_5, mgr._block_tables["r1"])
        mgr.free_request(state1)

        token_ids_8 = [1, 2, 3, 4, 5, 6, 7, 8]
        cached_len, cached_blocks = mgr.find_prefix_cache(token_ids_8)
        state2 = _make_state(token_ids_8, "r2")
        mgr.init_request_with_prefix(state2, cached_len, cached_blocks)
        state2.prefix_cached_len = cached_len
        state2.prefix_cached_blocks = list(cached_blocks)
        state2.prefilled = True  # 标记为已 prefill，触发 swap_out 路径

        mgr.swap_out(state2)

        # swap_out 后 prefix 字段应已清零
        assert state2.prefix_cached_len == 0
        assert state2.prefix_cached_blocks == []


# ---------------------------------------------------------------------------
# engine.generate 端到端（dry_run）
# ---------------------------------------------------------------------------

class TestEnginePrefixCacheE2E:
    def test_generate_completes_on_miss(self) -> None:
        """prefix cache miss 路径下，generate() 可以正常完成生成。"""
        engine = _make_engine()
        outputs = engine.generate(["hello world"], max_new_tokens=3)
        assert len(outputs) == 1
        assert engine.kv_cache.prefix_cache_size() > 0

    def test_generate_completes_on_hit(self) -> None:
        """prefix cache hit 路径下，generate() 可以正常完成生成，输出非空。"""
        engine = _make_engine()
        # 第一次：建立 prefix cache
        outputs1 = engine.generate(["hello world extra tokens"], max_new_tokens=2)
        assert len(outputs1) == 1
        cache_size = engine.kv_cache.prefix_cache_size()
        assert cache_size > 0

        # 第二次：相同前缀 + 不同后缀，应触发 prefix cache hit
        outputs2 = engine.generate(["hello world extra tokens more"], max_new_tokens=2)
        assert len(outputs2) == 1
        assert outputs2[0] != ""

    def test_prefix_cache_size_readable(self) -> None:
        """prefix_cache_size() 返回非负整数，generate 后大于等于 0。"""
        engine = _make_engine()
        engine.generate(["test"], max_new_tokens=2)
        size = engine.kv_cache.prefix_cache_size()
        assert isinstance(size, int)
        assert size >= 0

    def test_no_block_leak_after_hit(self) -> None:
        """prefix hit 请求完成后，free + prefix_cache_size 等于初始块数。"""
        engine = _make_engine(num_gpu_blocks=32, block_size=4)
        initial_free = engine.kv_cache.num_free_blocks()

        # miss：建立 prefix cache
        engine.generate(["hello world foo bar baz"], max_new_tokens=2)
        # hit：复用前缀
        engine.generate(["hello world foo bar baz qux"], max_new_tokens=2)

        assert (
            engine.kv_cache.num_free_blocks() + engine.kv_cache.prefix_cache_size()
            == initial_free
        )
