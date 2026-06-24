"""
Phase 3 / Phase 6 KV cache 测试。覆盖范围：
  - 块分配：init_request 按 prompt_len 分配正确数量的物理块
  - 块计数：num_free_blocks 随分配/释放正确变化
  - 块增长：decode 步骤超出当前块容量时自动分配新块
  - 块释放：free_request 归还所有物理块
  - 多请求隔离：两个请求各自占用独立的块，互不干扰
  - gather 正确性：向量化 gather_batch_kv 与手动计算结果一致（CPU，不依赖 GPU）
  - build_block_tables 正确性（Phase 6）：block_table/cache_seqlens 形状、dtype、数值正确

dry_run=True 的测试不依赖 GPU；gather 和 build_block_tables 正确性测试使用 device=cpu + dry_run=False。
"""

import torch

from mini_infer.core.config import EngineConfig
from mini_infer.cache.kv_cache import KVCacheManager
from mini_infer.core.request import Request, RequestState, SamplingParams


def _make_config(block_size: int = 4, num_gpu_blocks: int = 10) -> EngineConfig:
    return EngineConfig(
        model_name="stub",
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        dry_run=True,
    )


def build_state(prompt_len: int, request_id: str = "req-1") -> RequestState:
    return RequestState(
        request=Request(
            request_id=request_id,
            prompt="x" * prompt_len,
            sampling_params=SamplingParams(max_new_tokens=4),
        ),
        prompt_token_ids=[1] * prompt_len,
    )


def test_kv_cache_init_allocates_blocks() -> None:
    """init_request 应为 5 token 的 prompt 分配 ceil(5/4)=2 个块。"""
    manager = KVCacheManager(_make_config(block_size=4, num_gpu_blocks=10))
    state = build_state(prompt_len=5)

    assert manager.num_free_blocks() == 10
    manager.init_request(state)
    assert manager.num_free_blocks() == 8  # 2 blocks used

    alloc = manager.get_allocation(state.request.request_id)
    assert alloc is not None
    assert alloc.allocated_blocks == 2
    assert alloc.cached_tokens == 5


def test_kv_cache_free_returns_all_blocks() -> None:
    """free_request 应将所有物理块归还空闲池。"""
    manager = KVCacheManager(_make_config(block_size=4, num_gpu_blocks=10))
    state = build_state(prompt_len=5)

    manager.init_request(state)
    assert manager.num_free_blocks() == 8

    manager.free_request(state)
    assert manager.num_free_blocks() == 10
    assert manager.get_allocation(state.request.request_id) is None


def test_kv_cache_block_growth_on_decode() -> None:
    """write_decode_kv 在需要新块时应自动分配，块数随 token 增长而增加。"""
    manager = KVCacheManager(_make_config(block_size=4, num_gpu_blocks=10))
    state = build_state(prompt_len=5)  # blocks: ceil(5/4)=2, slots used: 0,1,2,3 in blk0; 0 in blk1

    manager.init_request(state)
    # seq_len=5: blk0 slots 0-3 full, blk1 slot 0 used

    # positions 5,6,7 → blk1 slots 1,2,3 → 仍在已分配的 2 个块内
    for _ in range(3):
        manager.write_decode_kv([state.request.request_id], None, None)
    assert manager.num_free_blocks() == 8  # still 2 blocks

    # position 8 → blk2 slot 0 → 需要第 3 个块
    manager.write_decode_kv([state.request.request_id], None, None)
    assert manager.num_free_blocks() == 7  # 3rd block allocated

    alloc = manager.get_allocation(state.request.request_id)
    assert alloc is not None
    assert alloc.allocated_blocks == 3
    assert alloc.cached_tokens == 9


def test_kv_cache_reservation_tracks_future_decode_blocks() -> None:
    """init_request 可预留未来 decode 块，准入判断应扣除未分配预留。"""
    manager = KVCacheManager(_make_config(block_size=4, num_gpu_blocks=10))
    state = build_state(prompt_len=5, request_id="r1")

    assert manager.can_reserve(6)
    manager.init_request(state, reserved_blocks=6)

    stats = manager.stats()
    assert stats["free_blocks"] == 8          # prompt 当前只实际分配 2 块
    assert stats["reserved_blocks"] == 6
    assert stats["outstanding_reserved_blocks"] == 4
    assert stats["reclaimable_prefix_blocks"] == 0
    assert stats["strict_available_blocks_for_reservation"] == 4
    assert stats["available_blocks_for_reservation"] == 4
    assert manager.can_reserve(4)
    assert not manager.can_reserve(5)

    manager.free_request(state)
    assert manager.stats()["strict_available_blocks_for_reservation"] == 10
    assert manager.stats()["available_blocks_for_reservation"] == 10


def test_kv_cache_two_requests_isolated() -> None:
    """两个请求各自持有独立的块，互不干扰；释放其一后另一个不受影响。"""
    manager = KVCacheManager(_make_config(block_size=4, num_gpu_blocks=10))
    s1 = build_state(prompt_len=5, request_id="r1")  # 2 blocks
    s2 = build_state(prompt_len=3, request_id="r2")  # ceil(3/4)=1 block

    manager.init_request(s1)
    manager.init_request(s2)
    assert manager.num_free_blocks() == 7  # 10 - 2 - 1

    manager.free_request(s1)
    assert manager.num_free_blocks() == 9  # r2 still holds 1 block

    alloc2 = manager.get_allocation("r2")
    assert alloc2 is not None
    assert alloc2.allocated_blocks == 1


def test_kv_cache_stats_tracks_usage() -> None:
    """stats 应返回 KV block 使用、请求占用和序列长度的只读快照。"""
    manager = KVCacheManager(_make_config(block_size=4, num_gpu_blocks=10))

    assert manager.stats() == {
        "total_blocks": 10,
        "free_blocks": 10,
        "used_blocks": 0,
        "utilization": 0.0,
        "active_requests": 0,
        "request_allocated_blocks": 0,
        "reserved_blocks": 0,
        "outstanding_reserved_blocks": 0,
        "reclaimable_prefix_blocks": 0,
        "strict_available_blocks_for_reservation": 10,
        "available_blocks_for_reservation": 10,
        "request_blocks": {},
        "request_reserved_blocks": {},
        "request_seq_lens": {},
        "prefix_cache_blocks": 0,
        "ref_counted_blocks": 0,
    }

    s1 = build_state(prompt_len=5, request_id="r1")  # 2 blocks
    s2 = build_state(prompt_len=3, request_id="r2")  # 1 block
    manager.init_request(s1)
    manager.init_request(s2)

    stats = manager.stats()
    assert stats["total_blocks"] == 10
    assert stats["free_blocks"] == 7
    assert stats["used_blocks"] == 3
    assert stats["utilization"] == 0.3
    assert stats["active_requests"] == 2
    assert stats["request_allocated_blocks"] == 3
    assert stats["reserved_blocks"] == 3
    assert stats["outstanding_reserved_blocks"] == 0
    assert stats["reclaimable_prefix_blocks"] == 0
    assert stats["strict_available_blocks_for_reservation"] == 7
    assert stats["available_blocks_for_reservation"] == 7
    assert stats["request_blocks"] == {"r1": 2, "r2": 1}
    assert stats["request_reserved_blocks"] == {"r1": 2, "r2": 1}
    assert stats["request_seq_lens"] == {"r1": 5, "r2": 3}
    assert stats["prefix_cache_blocks"] == 0
    assert stats["ref_counted_blocks"] == 3

    for _ in range(4):
        manager.write_decode_kv(["r1"], None, None)

    stats = manager.stats()
    assert stats["free_blocks"] == 6
    assert stats["used_blocks"] == 4
    assert stats["utilization"] == 0.4
    assert stats["request_allocated_blocks"] == 4
    assert stats["reserved_blocks"] == 4
    assert stats["outstanding_reserved_blocks"] == 0
    assert stats["reclaimable_prefix_blocks"] == 0
    assert stats["strict_available_blocks_for_reservation"] == 6
    assert stats["available_blocks_for_reservation"] == 6
    assert stats["request_blocks"] == {"r1": 3, "r2": 1}
    assert stats["request_reserved_blocks"] == {"r1": 3, "r2": 1}
    assert stats["request_seq_lens"] == {"r1": 9, "r2": 3}

    manager.free_request(s1)
    stats = manager.stats()
    assert stats["free_blocks"] == 9
    assert stats["used_blocks"] == 1
    assert stats["active_requests"] == 1
    assert stats["reserved_blocks"] == 1
    assert stats["outstanding_reserved_blocks"] == 0
    assert stats["reclaimable_prefix_blocks"] == 0
    assert stats["strict_available_blocks_for_reservation"] == 9
    assert stats["available_blocks_for_reservation"] == 9
    assert stats["request_blocks"] == {"r2": 1}
    assert stats["request_reserved_blocks"] == {"r2": 1}
    assert stats["request_seq_lens"] == {"r2": 3}


def test_kv_cache_no_free_blocks_raises() -> None:
    """空闲块耗尽时，init_request 应抛出 RuntimeError。"""
    manager = KVCacheManager(_make_config(block_size=4, num_gpu_blocks=2))
    s1 = build_state(prompt_len=8, request_id="r1")  # needs 2 blocks → uses all

    manager.init_request(s1)
    assert manager.num_free_blocks() == 0

    s2 = build_state(prompt_len=1, request_id="r2")
    try:
        manager.init_request(s2)
        assert False, "应该抛出 RuntimeError"
    except RuntimeError:
        pass


def test_gather_batch_kv_correctness() -> None:
    """
    向量化 gather_batch_kv 应正确聚合两个不同长度请求的 KV，并做左填充对齐。

    使用 device=cpu + dry_run=False，不依赖 GPU。
    配置：block_size=4, num_layers=1, num_kv_heads=1, head_dim=2
      - req_a: seq_len=6 → 2 块（block 0 存 token 0-3，block 1 存 token 4-5）
      - req_b: seq_len=3 → 1 块（block 2 存 token 0-2）
      - max_seq_len=6，req_b 有 3 个左填充位（应为 0）
    """
    config = EngineConfig(
        model_name="test",
        device="cpu",
        dtype="float32",
        num_gpu_blocks=8,
        block_size=4,
        num_hidden_layers=1,
        num_kv_heads=1,
        head_dim=2,
        dry_run=False,
    )
    manager = KVCacheManager(config)

    # 分配请求（block_size=4：req_a 需要 2 块，req_b 需要 1 块）
    sa = build_state(prompt_len=6, request_id="req_a")
    sb = build_state(prompt_len=3, request_id="req_b")
    manager.init_request(sa)
    manager.init_request(sb)

    # 确认块分配：req_a=[0,1], req_b=[2]（FreeBlockPool 按序分配）
    blk_a = manager._block_tables["req_a"]  # [0, 1]
    blk_b = manager._block_tables["req_b"]  # [2]
    assert len(blk_a) == 2
    assert len(blk_b) == 1

    # 写入已知 KV 值（layer 0）
    # k_cache[0] shape: [num_gpu_blocks=8, block_size=4, num_kv_heads=1, head_dim=2]
    # req_a token 0-3 → blk_a[0] slots 0-3
    for slot in range(4):
        manager.k_cache[0][blk_a[0], slot, 0, :] = torch.tensor([float(slot), float(slot + 10)])
    # req_a token 4-5 → blk_a[1] slots 0-1
    for slot in range(2):
        manager.k_cache[0][blk_a[1], slot, 0, :] = torch.tensor([float(slot + 4), float(slot + 14)])
    # req_b token 0-2 → blk_b[0] slots 0-2
    for slot in range(3):
        manager.k_cache[0][blk_b[0], slot, 0, :] = torch.tensor([float(slot + 20), float(slot + 30)])

    k_batch, _, seq_lens = manager.gather_batch_kv(["req_a", "req_b"])

    assert seq_lens == [6, 3]
    # k_batch[0] shape: [2, 1, 6, 2]
    assert k_batch[0].shape == (2, 1, 6, 2)

    # req_a（无填充）：output positions 0-5 = tokens 0-5
    expected_a = torch.tensor(
        [[[0, 10], [1, 11], [2, 12], [3, 13], [4, 14], [5, 15]]], dtype=torch.float32
    )  # [1, 6, 2]
    assert torch.allclose(k_batch[0][0], expected_a), f"req_a mismatch: {k_batch[0][0]}"

    # req_b（左填充 3 位）：output positions 0-2 = zeros，positions 3-5 = tokens 0-2
    expected_b = torch.tensor(
        [[[0, 0], [0, 0], [0, 0], [20, 30], [21, 31], [22, 32]]], dtype=torch.float32
    )  # [1, 6, 2]
    assert torch.allclose(k_batch[0][1], expected_b), f"req_b mismatch: {k_batch[0][1]}"


def test_build_block_tables() -> None:
    """
    Phase 6：build_block_tables 应返回正确 shape、dtype 和数值的 block_table / cache_seqlens。

    配置：block_size=4, num_gpu_blocks=8
      - req_a: prompt_len=6 → 2 块（块号 0, 1），seq_len=6
      - req_b: prompt_len=3 → 1 块（块号 2），seq_len=3
    期望：
      - block_table shape [2, 2]，dtype int32
        block_table[0] = [0, 1]，block_table[1] = [2, 0]（padding 填 0）
      - cache_seqlens shape [2]，dtype int32，值 [6, 3]
    """
    config = EngineConfig(
        model_name="test",
        device="cpu",
        dtype="float32",
        num_gpu_blocks=8,
        block_size=4,
        num_hidden_layers=1,
        num_kv_heads=1,
        head_dim=2,
        dry_run=False,
    )
    manager = KVCacheManager(config)

    sa = build_state(prompt_len=6, request_id="req_a")
    sb = build_state(prompt_len=3, request_id="req_b")
    manager.init_request(sa)
    manager.init_request(sb)

    # 确保下一个 decode slot 的块已分配（ensure_next_slot 预分配）
    manager.ensure_next_slot(["req_a", "req_b"])

    block_table, cache_seqlens = manager.build_block_tables(["req_a", "req_b"])

    # shape 和 dtype
    assert block_table.shape == (2, 2), f"block_table shape wrong: {block_table.shape}"
    assert block_table.dtype == torch.int32
    assert cache_seqlens.shape == (2,)
    assert cache_seqlens.dtype == torch.int32

    # cache_seqlens 值（seq_len = prompt_len，尚未 decode）
    assert cache_seqlens[0].item() == 6
    assert cache_seqlens[1].item() == 3

    # block_table 数值：req_a 占块 0,1；req_b 占块 2（ensure 后 req_b 的 block_idx=3//4=0，已有）
    blk_a = manager._block_tables["req_a"]
    blk_b = manager._block_tables["req_b"]
    assert block_table[0, 0].item() == blk_a[0]
    assert block_table[0, 1].item() == blk_a[1]
    assert block_table[1, 0].item() == blk_b[0]
    assert block_table[1, 1].item() == 0  # padding


def test_ensure_next_slot_allocates_new_block() -> None:
    """
    Phase 6：ensure_next_slot 在 token 恰好到达新块起点时分配新块。

    配置：block_size=4，prompt_len=4 → 1 块（slots 0-3 全满）
    seq_len=4 时下一个 decode 位置 block_idx=1，需要新块。
    """
    config = EngineConfig(
        model_name="test",
        device="cpu",
        dtype="float32",
        num_gpu_blocks=8,
        block_size=4,
        num_hidden_layers=1,
        num_kv_heads=1,
        head_dim=2,
        dry_run=False,
    )
    manager = KVCacheManager(config)
    sa = build_state(prompt_len=4, request_id="req_a")
    manager.init_request(sa)

    assert len(manager._block_tables["req_a"]) == 1
    assert manager.num_free_blocks() == 7

    # seq_len=4, block_idx=4//4=1 → 超出已分配块数，需要新块
    manager.ensure_next_slot(["req_a"])
    assert len(manager._block_tables["req_a"]) == 2
    assert manager.num_free_blocks() == 6


def test_advance_seq_lens() -> None:
    """Phase 6：advance_seq_lens 将各请求 seq_len 递增 1。"""
    manager = KVCacheManager(_make_config(block_size=4, num_gpu_blocks=8))
    sa = build_state(prompt_len=5, request_id="req_a")
    sb = build_state(prompt_len=3, request_id="req_b")
    manager.init_request(sa)
    manager.init_request(sb)

    assert manager._seq_lens["req_a"] == 5
    assert manager._seq_lens["req_b"] == 3

    manager.advance_seq_lens(["req_a", "req_b"])

    assert manager._seq_lens["req_a"] == 6
    assert manager._seq_lens["req_b"] == 4
