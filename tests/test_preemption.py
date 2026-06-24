"""tests/test_preemption.py — Phase 7 Preemption + Priority Scheduling 测试。

覆盖范围（全部 dry_run=True，无需 GPU）：

KVCacheManager 接口：
  - swap_out 后 GPU 块全部释放，swapped_seq_len 已记录
  - swap_in 后块重新分配，seq_len 恢复，cpu_kv 清除
  - swap_out → swap_in 后 seq_len 与原始一致

Scheduler 接口：
  - mark_swapped 将请求从 running 移到 swapped
  - un_admit 将刚准入未 prefill 的请求移回 waiting 队尾（不进入 swapped）
  - get_lowest_priority_running 返回优先级数值最大的请求
  - move_swapped_to_running 将请求从 swapped 移到 running
  - has_swapped / num_swapped 计数正确

LLMEngine 端到端（dry_run）：
  - KV 不足时触发抢占而非报错：
    · 未 prefill 的低优先级请求：un_admit（撤销准入，回 waiting，不进入 swapped）
    · 已 prefill 的低优先级请求：swap_out（KV 换到 CPU，进入 swapped 队列）
  - 被抢占请求最终完成生成
  - 高优先级请求先完成
  - 相同优先级不触发抢占
  - generate(priorities=...) 参数长度不匹配时报 ValueError
  - 现有 test_engine.py 中 test_kv_exhaustion 场景：相同优先级无法抢占时仍然报错
  - 注：已 prefill 请求的真实 swap 场景（GPU-to-CPU KV 拷贝语义）在 GPU 实机验证
"""

import math
import pytest

from mini_infer import EngineConfig, LLMEngine
from mini_infer.cache.kv_cache import KVCacheManager
from mini_infer.core.request import Request, RequestState, SamplingParams
from mini_infer.runtime.scheduler import Scheduler


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def _make_config(
    block_size: int = 4,
    num_gpu_blocks: int = 16,
    max_batch_size: int = 4,
) -> EngineConfig:
    return EngineConfig(
        model_name="stub",
        dry_run=True,
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        max_batch_size=max_batch_size,
    )


def _make_state(
    prompt_len: int = 4,
    max_new_tokens: int = 4,
    priority: int = 0,
    request_id: str = "req-1",
) -> RequestState:
    return RequestState(
        request=Request(
            request_id=request_id,
            prompt="x" * prompt_len,
            sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
            priority=priority,
        ),
        prompt_token_ids=[1] * prompt_len,
    )


def _make_engine(
    num_gpu_blocks: int = 32,
    block_size: int = 4,
    max_batch_size: int = 4,
    scheduler_policy: str = "pressure_aware",
    scheduler_ttft_slo_steps: int = 16,
) -> LLMEngine:
    config = EngineConfig(
        model_name="stub",
        dry_run=True,
        block_size=block_size,
        num_gpu_blocks=num_gpu_blocks,
        max_batch_size=max_batch_size,
        scheduler_policy=scheduler_policy,
        scheduler_ttft_slo_steps=scheduler_ttft_slo_steps,
    )
    return LLMEngine(config)


# ---------------------------------------------------------------------------
# KVCacheManager: swap_out
# ---------------------------------------------------------------------------

class TestSwapOut:
    def test_swap_out_frees_gpu_blocks(self) -> None:
        """swap_out 后，请求的 GPU 块全部归还到空闲池。"""
        config = _make_config(block_size=4, num_gpu_blocks=10)
        mgr = KVCacheManager(config)
        state = _make_state(prompt_len=5)  # 需要 ceil(5/4)=2 块

        mgr.init_request(state)
        free_after_init = mgr.num_free_blocks()  # 10 - 2 = 8

        mgr.swap_out(state)
        assert mgr.num_free_blocks() == free_after_init + 2  # 2 块归还

    def test_swap_out_records_seq_len(self) -> None:
        """swap_out 将当前 seq_len 记录到 state.swapped_seq_len。"""
        config = _make_config(block_size=4, num_gpu_blocks=10)
        mgr = KVCacheManager(config)
        state = _make_state(prompt_len=5)

        mgr.init_request(state)
        assert mgr._seq_lens[state.request.request_id] == 5

        mgr.swap_out(state)
        assert state.swapped_seq_len == 5

    def test_swap_out_removes_request_from_metadata(self) -> None:
        """swap_out 后，block_table 和 seq_len 元数据已清除。"""
        config = _make_config(block_size=4, num_gpu_blocks=10)
        mgr = KVCacheManager(config)
        state = _make_state(prompt_len=5)

        mgr.init_request(state)
        mgr.swap_out(state)

        assert state.request.request_id not in mgr._block_tables
        assert state.request.request_id not in mgr._seq_lens


# ---------------------------------------------------------------------------
# KVCacheManager: swap_in
# ---------------------------------------------------------------------------

class TestSwapIn:
    def test_swap_in_reallocates_blocks(self) -> None:
        """swap_in 后，块重新分配，数量等于 ceil(swapped_seq_len / block_size)。"""
        config = _make_config(block_size=4, num_gpu_blocks=10)
        mgr = KVCacheManager(config)
        state = _make_state(prompt_len=5)

        mgr.init_request(state)
        mgr.swap_out(state)

        free_before = mgr.num_free_blocks()  # == 10（全部归还）
        mgr.swap_in(state)
        # swapped_seq_len=5 → ceil(5/4)=2 块
        assert mgr.num_free_blocks() == free_before - 2

    def test_swap_in_restores_seq_len(self) -> None:
        """swap_in 后，_seq_lens 恢复为 swapped_seq_len。"""
        config = _make_config(block_size=4, num_gpu_blocks=10)
        mgr = KVCacheManager(config)
        state = _make_state(prompt_len=6)  # ceil(6/4)=2 块

        mgr.init_request(state)
        mgr.swap_out(state)
        mgr.swap_in(state)

        assert mgr._seq_lens[state.request.request_id] == 6

    def test_swap_in_clears_cpu_kv(self) -> None:
        """swap_in 后，state.cpu_kv 置 None（dry_run 下本来就是 None，此处验证字段行为）。"""
        config = _make_config(block_size=4, num_gpu_blocks=10)
        mgr = KVCacheManager(config)
        state = _make_state(prompt_len=4)

        mgr.init_request(state)
        mgr.swap_out(state)
        mgr.swap_in(state)

        assert state.cpu_kv is None

    def test_swap_out_then_swap_in_idempotent_seq_len(self) -> None:
        """swap_out → swap_in 后，seq_len 与初始 init_request 的值相同。"""
        config = _make_config(block_size=4, num_gpu_blocks=10)
        mgr = KVCacheManager(config)
        state = _make_state(prompt_len=7)  # ceil(7/4)=2 块

        mgr.init_request(state)
        original_seq_len = mgr._seq_lens[state.request.request_id]

        mgr.swap_out(state)
        mgr.swap_in(state)

        assert mgr._seq_lens[state.request.request_id] == original_seq_len


# ---------------------------------------------------------------------------
# Scheduler: preemption 接口
# ---------------------------------------------------------------------------

class TestSchedulerPreemption:
    def test_mark_swapped_moves_from_running(self) -> None:
        sched = Scheduler(max_batch_size=4)
        state = _make_state()
        sched._running[state.request.request_id] = state

        sched.mark_swapped(state)

        assert state.request.request_id not in sched._running
        assert sched.has_swapped()
        assert sched.num_swapped() == 1

    def test_get_lowest_priority_running_empty(self) -> None:
        sched = Scheduler(max_batch_size=4)
        assert sched.get_lowest_priority_running() is None

    def test_get_lowest_priority_running_single(self) -> None:
        sched = Scheduler(max_batch_size=4)
        state = _make_state(priority=5)
        sched._running[state.request.request_id] = state

        assert sched.get_lowest_priority_running() is state

    def test_get_lowest_priority_running_multiple(self) -> None:
        sched = Scheduler(max_batch_size=4)
        high = _make_state(priority=1, request_id="high")
        low = _make_state(priority=9, request_id="low")
        mid = _make_state(priority=5, request_id="mid")
        for s in (high, low, mid):
            sched._running[s.request.request_id] = s

        victim = sched.get_lowest_priority_running()
        assert victim is low  # priority=9 最大 → 优先级最低

    def test_move_swapped_to_running(self) -> None:
        sched = Scheduler(max_batch_size=4)
        state = _make_state()
        sched._swapped.append(state)

        sched.move_swapped_to_running(state)

        assert not sched.has_swapped()
        assert state.request.request_id in sched._running

    def test_has_swapped_false_when_empty(self) -> None:
        sched = Scheduler(max_batch_size=4)
        assert not sched.has_swapped()
        assert sched.num_swapped() == 0

    def test_un_admit_returns_to_waiting(self) -> None:
        """un_admit 将请求从 running 移回 waiting 队尾（不进入 _swapped）。"""
        sched = Scheduler(max_batch_size=4)
        state = _make_state()
        sched.add_request(state)          # 先入 waiting
        sched.pop_next_waiting()          # 模拟准入：从 waiting 取出
        sched.add_to_running(state)       # 加入 running

        sched.un_admit(state)

        assert state.request.request_id not in sched._running  # 从 running 移除
        assert not sched.has_swapped()                         # 不进入 swapped
        assert sched.has_waiting()                             # 回到 waiting
        assert sched._waiting[-1] is state                     # 放在队尾

    def test_scheduler_stats_tracks_counters(self) -> None:
        """Scheduler.stats 应统计准入、抢占、换出、换入、完成和拒绝次数。"""
        sched = Scheduler(max_batch_size=4)
        state = _make_state()
        sched.add_to_running(state)
        sched.mark_swapped(state)
        sched.move_swapped_to_running(state)
        sched.finish_request(state)
        sched.record_reject()

        stats = sched.stats()
        assert stats["waiting"] == 0
        assert stats["running"] == 0
        assert stats["swapped"] == 0
        assert stats["counters"] == {
            "admit_count": 1,
            "reject_count": 1,
            "preempt_count": 1,
            "un_admit_count": 0,
            "swap_out_count": 1,
            "swap_in_count": 1,
            "finish_count": 1,
        }


# ---------------------------------------------------------------------------
# LLMEngine: 端到端抢占场景
# ---------------------------------------------------------------------------

class TestEnginePreemption:
    def test_preemption_on_kv_exhaustion(self) -> None:
        """
        KV 块不足 + 有低优先级 running 请求时，触发换出而非 RuntimeError。

        设置：block_size=4，num_gpu_blocks=4
          - 请求 A（低优先级=10）：prompt "ab"=2 token + max_new_tokens=2 → 需要 ceil(4/4)=1 块
          - 请求 B（高优先级=0）：同样参数 → 1 块
          - A 先加入，占用 1 块；B 后加入时若还剩 0 块，触发换出 A
        """
        # num_gpu_blocks=2：A 和 B 各需要 1 块，但都有 max_new_tokens=2 步 decode，
        # 加上 prompt 共 4 token = 1 块刚好（block_size=4）
        # 为触发换出：设 num_gpu_blocks=1，A 的 1 块占满后 B 无法准入
        engine = _make_engine(num_gpu_blocks=1, block_size=4, max_batch_size=2)
        # "ab" → 2 tokens；max_new_tokens=1；blocks_needed=ceil((2+1)/4)=1 → 刚好 1 块
        # A 先入，占 1 块；B 等 → 块不足，A 优先级 10 > B 优先级 0 → 换出 A
        outputs = engine.generate(["ab", "cd"], max_new_tokens=1, priorities=[10, 0])
        assert len(outputs) == 2
        # 两个请求最终都完成
        for out in outputs:
            assert isinstance(out, str)

    def test_swapped_request_eventually_completes(self) -> None:
        """被换出的请求在 GPU 空间恢复后，最终完成生成（不会永久挂起）。"""
        engine = _make_engine(num_gpu_blocks=2, block_size=4, max_batch_size=2)
        # "ab" = 2 tokens, max_new_tokens=2, blocks_needed=ceil(4/4)=1 块
        # 2 块可以同时容纳 2 个请求，但加 max_batch_size=2 时若 num_gpu_blocks 很小可能触发
        # 用 num_gpu_blocks=1 强制换出
        engine2 = _make_engine(num_gpu_blocks=1, block_size=4, max_batch_size=4)
        outputs = engine2.generate(["ab", "cd"], max_new_tokens=1, priorities=[10, 0])
        assert len(outputs) == 2
        assert engine2.kv_cache.num_free_blocks() == 1  # 所有块归还

    def test_same_priority_no_preemption_raises(self) -> None:
        """相同优先级时，无法抢占，KV 耗尽仍然报 RuntimeError。"""
        # 复现原 test_kv_exhaustion 场景，但用相同优先级（不触发抢占）
        engine = _make_engine(num_gpu_blocks=1, block_size=4)
        with pytest.raises(RuntimeError, match="KV 块"):
            # 两个请求优先级相同，后者不能换出前者
            engine.generate(["ab", "cd"], max_new_tokens=4, priorities=[0, 0])

    def test_same_priority_short_request_can_preempt_large_running_once(self) -> None:
        """同优先级短请求可抢占一次大 KV 请求，但不能形成活锁。"""
        engine = _make_engine(num_gpu_blocks=4, block_size=4, max_batch_size=4)
        engine.add_request("L" * 12, max_new_tokens=4, priority=0, request_id="long")

        # 先让 long 完成 prefill 并进入 running，占满 4 个预留块。
        engine.step()
        for rid, prompt in [("s1", "a"), ("s2", "b"), ("s3", "c")]:
            engine.add_request(prompt, max_new_tokens=2, priority=0, request_id=rid)

        for _ in range(16):
            if not engine.has_unfinished_requests():
                break
            engine.step()

        assert not engine.has_unfinished_requests()
        counters = engine.scheduler.stats()["counters"]
        assert counters["finish_count"] == 4
        assert counters["preempt_count"] == 1
        assert counters["swap_out_count"] == 1
        assert counters["swap_in_count"] == 1

    def test_slo_aware_waiting_selection_prefers_high_priority(self) -> None:
        """slo_aware 不被 waiting FIFO 队头阻塞，优先准入高优先级请求。"""
        engine = _make_engine(
            num_gpu_blocks=8,
            block_size=4,
            max_batch_size=1,
            scheduler_policy="slo_aware",
        )
        low_id = engine.add_request("L" * 8, max_new_tokens=4, priority=5, request_id="low")
        high_id = engine.add_request("H" * 4, max_new_tokens=2, priority=0, request_id="high")

        tokens = engine.step()

        assert high_id in tokens
        assert low_id not in tokens
        assert engine._step_states[high_id].admit_step == 1
        assert engine._step_states[low_id].admit_step is None

    def test_slo_aware_same_priority_waiter_can_preempt_after_ttft_slo_risk(self) -> None:
        """同优先级请求等待超过 TTFT SLO 后，可以抢占未接近完成的长请求。"""
        engine = _make_engine(
            num_gpu_blocks=4,
            block_size=4,
            max_batch_size=4,
            scheduler_policy="slo_aware",
            scheduler_ttft_slo_steps=1,
        )
        engine.add_request("L" * 8, max_new_tokens=8, priority=0, request_id="long")

        # long 完成 prefill 并进入 running，占满 4 个预留块。
        engine.step()
        engine.add_request("S" * 4, max_new_tokens=4, priority=0, request_id="short")

        # short 已经等待 1 step，达到 TTFT SLO risk；下一步应抢占 long。
        engine.step()

        counters = engine.scheduler.stats()["counters"]
        assert counters["preempt_count"] == 1
        assert counters["swap_out_count"] == 1
        assert engine.scheduler.has_swapped()

    def test_priorities_length_mismatch_raises(self) -> None:
        """priorities 长度与 prompts 不一致时，立即报 ValueError。"""
        engine = _make_engine()
        with pytest.raises(ValueError, match="priorities 长度"):
            engine.generate(["a", "b"], max_new_tokens=2, priorities=[1])

    def test_generate_without_priorities_backward_compatible(self) -> None:
        """不传 priorities 时行为与 Phase 2 完全一致（所有请求优先级=0）。

        Phase 10：prefix cache 可能持有部分 blocks，因此检查 free + cache_size == total。
        """
        engine = _make_engine(num_gpu_blocks=32)
        outputs = engine.generate(["hello", "world"], max_new_tokens=3)
        assert len(outputs) == 2
        assert engine.kv_cache.num_free_blocks() + engine.kv_cache.prefix_cache_size() == 32

    def test_high_priority_request_preempts_low(self) -> None:
        """高优先级（priority=0）请求在 KV 不足时，成功换出低优先级（priority=5）请求。"""
        engine = _make_engine(num_gpu_blocks=1, block_size=4, max_batch_size=2)
        # priority=5 的先进入等待，priority=0 的后进入等待
        # 实际 generate 会先准入 priority=5，然后 priority=0 来时块不足 → 换出 priority=5
        outputs = engine.generate(["ab", "xy"], max_new_tokens=1, priorities=[5, 0])
        assert len(outputs) == 2
        assert engine.kv_cache.num_free_blocks() == 1  # 所有块归还
        counters = engine.scheduler.stats()["counters"]
        assert counters["preempt_count"] >= 1
        assert counters["un_admit_count"] >= 1

    def test_never_prefilled_request_is_unadmitted_not_swapped(self) -> None:
        """
        刚准入但尚未 prefill 的低优先级请求被抢占时，走 un_admit 路径（不进入 _swapped）。

        验证修复：blocking issue —— 之前 swap_out 会清空 KV 元数据导致 GPU 路径 KeyError。
        un_admit 路径的正确行为：请求回到 waiting 队尾，GPU 块正常归还，不出现在 swapped 中。
        """
        engine = _make_engine(num_gpu_blocks=1, block_size=4, max_batch_size=2)
        # 低优先级请求先进 waiting，高优先级后进 waiting
        # 准入循环：先 admit 低优先级（未 prefill），高优先级来时触发 un_admit
        outputs = engine.generate(["ab", "cd"], max_new_tokens=1, priorities=[10, 0])
        assert len(outputs) == 2
        # 关键验证：un_admit 路径不产生 swapped 请求
        assert not engine.scheduler.has_swapped()
        # 所有块归还
        assert engine.kv_cache.num_free_blocks() == 1
        counters = engine.scheduler.stats()["counters"]
        assert counters["preempt_count"] >= 1
        assert counters["un_admit_count"] >= 1
