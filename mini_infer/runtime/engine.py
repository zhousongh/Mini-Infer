"""
Phase 7/8/9/10 推理引擎。

Phase 7：continuous batching + preemption + priority scheduling（generate() 接口）。

Phase 8 新增：单步接口，供 AsyncEngine / HTTP 服务使用：
  - add_request(prompt, max_new_tokens, priority) → request_id
      将单个请求加入等待队列，返回 request_id。
  - step() → {request_id: [new_token_texts]}
      执行一次 prefill + decode_batch，返回本步每个请求新生成的 token 文本列表。
  - has_unfinished_requests() → bool
  - is_finished(request_id) → bool

Phase 9 新增：Chunked Prefill（chunk_prefill_size > 0 时启用）：
  - 长 prompt 请求分多步 prefill，每步只处理 chunk_prefill_size 个 token
  - PREFILLING 状态介于 WAITING 和 RUNNING 之间，一次最多 1 个请求处于该状态
  - 中间 DynamicCache 保存在 _prefilling_caches（当前实现保留在模型设备上），最后一个 chunk 后写入 block tensor
  - generate() 和 step() 两条路径均支持，chunk_prefill_size=0 时行为与 Phase 8 完全一致

注意：generate() 和 add_request/step() 使用同一个 scheduler/kv_cache，
不得同时混用——同一时刻只用一种接口。

主循环结构（每次 step 迭代）：
  1. 准入：接入尽可能多的等待请求，块不足时尝试抢占低优先级 running 请求
     （chunk_prefill_size > 0 时：准入到 PREFILLING 而非直接 RUNNING）
  2. Prefill：对新接入的请求做 prefill（或推进当前 PREFILLING 请求的一个 chunk）
  3. Batch decode：一次 forward 处理所有 running 请求
  4. 清理：移除已完成请求，释放 KV 块
  5. Swap_in：有空闲块时将 swapped 请求换回，加入 running
"""

import json
import math
import os
import time
from uuid import uuid4

from ..cache.kv_cache import KVCacheManager
from ..core.config import EngineConfig
from ..core.request import Request, RequestState, SamplingParams
from ..modeling.model_runner import ModelRunner
from .scheduler import Scheduler


class LLMEngine:
    """提供 generate 接口，实现 continuous batching + preemption 推理调度。"""

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        if not config.dry_run and config.block_size % 256 != 0:
            raise ValueError(
                "真实 LLMEngine decode 路径使用 flash_attn_with_kvcache，"
                f"block_size 必须是 256 的倍数，当前为 {config.block_size}。"
            )
        self.kv_cache = KVCacheManager(config=config)
        self.scheduler = Scheduler(max_batch_size=config.max_batch_size)
        self.model_runner = ModelRunner(config=config, kv_cache=self.kv_cache)
        # Phase 8：单步接口的请求状态追踪（request_id → RequestState）
        self._step_states: dict[str, RequestState] = {}
        # Phase 9：chunked prefill 时保存中间 DynamicCache（request_id → DynamicCache | None）
        self._prefilling_caches: dict[str, object] = {}
        self._same_priority_preempted: set[str] = set()
        self._kv_trace_file = os.getenv("MINI_INFER_KV_TRACE_FILE", "")
        self._step_idx = 0
        if self._kv_trace_file:
            trace_dir = os.path.dirname(self._kv_trace_file)
            if trace_dir:
                os.makedirs(trace_dir, exist_ok=True)
        # Phase 12：CUDA Graph warmup（config.use_cuda_graph=True 时触发）
        if config.use_cuda_graph and not config.dry_run:
            self.model_runner.warmup_cuda_graphs()

    def _scheduler_stats(self) -> dict[str, object]:
        stats = self.scheduler.stats()
        stats.update({
            "running_ids": [
                state.request.request_id for state in self.scheduler.get_running_states()
            ],
            "prefilling_ids": [
                state.request.request_id for state in self.scheduler.get_prefilling_states()
            ],
            "swapped_ids": [
                state.request.request_id for state in self.scheduler.get_swapped_states()
            ],
        })
        return stats

    def _trace_kv(self, event: str, **extra: object) -> None:
        """按 JSONL 记录 KV cache / scheduler 状态；未配置时无开销返回。"""
        if not self._kv_trace_file:
            return
        record: dict[str, object] = {
            "ts": time.time(),
            "step": self._step_idx,
            "event": event,
            "scheduler": self._scheduler_stats(),
            "kv_cache": self.kv_cache.stats(),
        }
        if extra:
            record.update(extra)
        try:
            with open(self._kv_trace_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            # Trace 是调试辅助，不应影响线上推理路径。
            pass

    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int = 128,
        priorities: list[int] | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> list[str]:
        """
        批量生成，返回与输入 prompts 顺序一致的输出文本列表。

        priorities: 每个 prompt 的调度优先级（0 = 最高，数值越大优先级越低）。
                    若为 None，所有请求优先级为 0（与 Phase 2 行为一致）。
        """
        if priorities is not None and len(priorities) != len(prompts):
            raise ValueError(f"priorities 长度 {len(priorities)} 与 prompts 长度 {len(prompts)} 不一致")

        states: list[RequestState] = []
        for i, prompt in enumerate(prompts):
            priority = priorities[i] if priorities is not None else 0
            request = Request(
                request_id=str(uuid4()),
                prompt=prompt,
                sampling_params=SamplingParams(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                ),
                priority=priority,
            )
            state = RequestState(
                request=request,
                prompt_token_ids=self._tokenize(prompt),
                arrival_step=self._step_idx,
            )
            self.scheduler.add_request(state)
            self._trace_kv(
                "add_request_generate",
                request_id=state.request.request_id,
                prompt_len=len(state.prompt_token_ids),
                max_new_tokens=max_new_tokens,
                priority=priority,
            )
            states.append(state)

        outputs: dict[str, str] = {}

        try:
            while (
                self.scheduler.has_waiting()
                or self.scheduler.has_prefilling()
                or self.scheduler.num_running() > 0
                or self.scheduler.has_swapped()
            ):
                if self.config.chunk_prefill_size > 0:
                    # ── Chunked Prefill 路径（Phase 9）────────────────────────
                    # 1a. 若无正在进行的 prefill，从 waiting 准入一个请求到 PREFILLING
                    if (
                        not self.scheduler.has_prefilling()
                        and self.scheduler.has_waiting()
                        and (self.scheduler.num_running() + self.scheduler.num_prefilling())
                            < self.config.max_batch_size
                    ):
                        next_state = self._peek_next_waiting_state()
                        assert next_state is not None
                        prompt_len = len(next_state.prompt_token_ids)
                        max_out = next_state.request.sampling_params.max_new_tokens
                        blocks_needed = math.ceil((prompt_len + max_out) / self.config.block_size)
                        if blocks_needed > self.config.num_gpu_blocks:
                            self.scheduler.record_reject()
                            raise RuntimeError(
                                f"请求 {next_state.request.request_id!r} 需要 {blocks_needed} 个 KV 块"
                                f"（prompt={prompt_len} + max_new_tokens={max_out}），"
                                f"超过系统总块数 {self.config.num_gpu_blocks}。"
                            )
                        if self._can_admit_blocks(blocks_needed):
                            state = self._pop_waiting_state(next_state)
                            self._init_request_for_policy(state, blocks_needed)
                            self.scheduler.add_to_prefilling(state)
                        else:
                            victim = self._select_victim(next_state, blocks_needed)
                            if victim is not None:
                                self._preempt_victim(victim, requester=next_state)
                                # 腾出块后立即尝试准入
                                if self._can_admit_blocks(blocks_needed):
                                    state = self._pop_waiting_state(next_state)
                                    self._init_request_for_policy(state, blocks_needed)
                                    self.scheduler.add_to_prefilling(state)

                    # 2a. 推进当前 PREFILLING 请求的一个 chunk
                    pf_state = self.scheduler.get_next_prefilling()
                    if pf_state is not None:
                        rid = pf_state.request.request_id
                        t_start = pf_state.prefilled_tokens
                        t_end = min(
                            t_start + self.config.chunk_prefill_size,
                            len(pf_state.prompt_token_ids),
                        )
                        is_last = (t_end == len(pf_state.prompt_token_ids))
                        cache = self._prefilling_caches.get(rid)
                        new_cache = self.model_runner.prefill_chunk(
                            pf_state, t_start, t_end, cache, is_last
                        )
                        if is_last:
                            self._prefilling_caches.pop(rid, None)
                            self.scheduler.move_prefilling_to_running(pf_state)
                        else:
                            self._prefilling_caches[rid] = new_cache

                else:
                    # ── 原始路径（Phase 8 行为，chunk_prefill_size == 0）────────
                    # Phase 10：此路径支持 prefix cache（chunked prefill 路径不走 prefix cache）
                    # 1. 准入：接入尽可能多的等待请求，块不足时尝试抢占
                    newly_admitted: list[RequestState] = []
                    while (
                        self.scheduler.has_waiting()
                        and self.scheduler.num_running() < self.config.max_batch_size
                    ):
                        next_state = self._peek_next_waiting_state()
                        assert next_state is not None

                        prompt_len = len(next_state.prompt_token_ids)
                        max_out = next_state.request.sampling_params.max_new_tokens
                        # Phase 10：预先查询 prefix cache，仅对 suffix + decode 部分计算所需新块数。
                        # cached_len > 0 时，这些块已在 cache 中，不需要新分配，避免准入时过度保守。
                        cached_len_peek, _ = self.kv_cache.find_prefix_cache(
                            next_state.prompt_token_ids
                        )
                        suffix_len = prompt_len - cached_len_peek  # no hit → suffix_len = prompt_len
                        blocks_needed = math.ceil((suffix_len + max_out) / self.config.block_size)

                        if blocks_needed > self.config.num_gpu_blocks:
                            self.scheduler.record_reject()
                            raise RuntimeError(
                                f"请求 {next_state.request.request_id!r} 需要 {blocks_needed} 个 KV 块"
                                f"（suffix={suffix_len} + max_new_tokens={max_out}，"
                                f"prefix_cached={cached_len_peek} tokens），"
                                f"超过系统总块数 {self.config.num_gpu_blocks}。"
                                f"请增大 num_gpu_blocks 或减小 max_new_tokens。"
                            )

                        if self._can_admit_blocks(blocks_needed):
                            state = self._pop_waiting_state(next_state)
                            self._admit_with_prefix(state, blocks_needed)  # Phase 10：prefix cache 感知准入
                            self.scheduler.add_to_running(state)
                            newly_admitted.append(state)
                        else:
                            victim = self._select_victim(next_state, blocks_needed)
                            if victim is not None:
                                self._preempt_victim(victim, newly_admitted, requester=next_state)
                                continue

                            if self.scheduler.num_running() == 0 and not newly_admitted:
                                self.scheduler.record_reject()
                                raise RuntimeError(
                                    f"请求 {next_state.request.request_id!r} 需要 {blocks_needed} 个 KV 块"
                                    f"（suffix={suffix_len} + max_new_tokens={max_out}），"
                                    f"但当前仅有 {self.kv_cache.num_free_blocks()} 个空闲块，"
                                    f"且没有优先级更低的 running 请求可换出。"
                                )
                            break

                    # 2. Prefill（Phase 10：prefix hit 请求走 prefill_with_prefix，并在完成后注册缓存）
                    if newly_admitted:
                        self._prefill_and_register(newly_admitted)

                # ── 3. Batch decode：两条路径共用 ────────────────────────────
                running = self.scheduler.get_running_states()
                if running:
                    self.model_runner.decode_batch(running)

                # ── 4. 清理完成的请求 ────────────────────────────────────────
                for state in list(running):
                    if state.finished:
                        state.decoded_text = self.model_runner.tokenizer.decode(
                            state.generated_token_ids, skip_special_tokens=True
                        )
                        outputs[state.request.request_id] = state.decoded_text
                        self.kv_cache.free_request(state)
                        self.scheduler.finish_request(state)

                # ── 5. 换入：将 swapped 请求换回 GPU（有足够块时）────────────
                # 注意：检查时同时计入 num_prefilling()，防止 PREFILLING 请求完成后
                # 导致 running + 1 超出 max_batch_size（chunked prefill 场景下的边界条件）
                for swapped_state in self.scheduler.get_swapped_states():
                    if (self.scheduler.num_running() + self.scheduler.num_prefilling()
                            >= self.config.max_batch_size):
                        break
                    needed = self._reserved_blocks_for_swapped(swapped_state)
                    if self._can_swap_in(swapped_state):
                        self._swap_in_for_policy(swapped_state)
                        self.scheduler.move_swapped_to_running(swapped_state)
                    else:
                        break  # FIFO：第一个换不回来则停止

        except Exception:
            # 异常时归还所有 running 请求的 GPU 块，防止显存泄漏
            for state in self.scheduler.get_running_states():
                try:
                    self.kv_cache.free_request(state)
                except Exception as free_exc:
                    import sys
                    print(
                        f"warning: free_request failed for {state.request.request_id!r}: {free_exc}",
                        file=sys.stderr,
                    )
            # Phase 9：归还 PREFILLING 请求的 GPU 块
            for state in self.scheduler.get_prefilling_states():
                try:
                    self.kv_cache.free_request(state)
                except Exception as free_exc:
                    import sys
                    print(
                        f"warning: free_request failed for {state.request.request_id!r}: {free_exc}",
                        file=sys.stderr,
                    )
            self._prefilling_caches.clear()
            # 清除 swapped 请求的 CPU KV，释放 CPU 内存
            for state in self.scheduler.get_swapped_states():
                state.cpu_kv = None
            raise

        return [outputs.get(state.request.request_id, "") for state in states]

    def _tokenize(self, text: str) -> list[int]:
        return self.model_runner.tokenizer.encode(text, add_special_tokens=True)

    def _waiting_blocks_estimate(self, state: RequestState) -> int:
        total_tokens = (
            len(state.prompt_token_ids)
            + state.request.sampling_params.max_new_tokens
        )
        return max(1, math.ceil(total_tokens / self.config.block_size))

    def _request_wait_steps(self, state: RequestState) -> int:
        return max(0, self._step_idx - state.arrival_step)

    def _request_slo_risk(self, state: RequestState) -> int:
        wait = self._request_wait_steps(state)
        if wait >= self.config.scheduler_ttft_slo_steps:
            return 2
        if wait >= max(1, self.config.scheduler_ttft_slo_steps // 2):
            return 1
        return 0

    def _waiting_priority_score(self, state: RequestState) -> tuple[int, int, int, int, int, int]:
        """
        SLO-aware waiting selection score.

        Bigger tuple wins:
          1. higher user priority (priority value is smaller)
          2. higher TTFT SLO risk
          3. smaller KV footprint
          4. smaller requested decode length
          5. longer waiting time
          6. older arrival as tie-breaker

        TTFT is only one serving-quality signal. This ordering keeps high
        priority and SLO-risk handling, but avoids letting an aged large request
        repeatedly jump ahead of short requests that can finish quickly.
        """
        max_new = state.request.sampling_params.max_new_tokens
        return (
            -state.request.priority,
            self._request_slo_risk(state),
            -self._waiting_blocks_estimate(state),
            -max_new,
            self._request_wait_steps(state),
            -state.arrival_step,
        )

    def _peek_next_waiting_state(self) -> RequestState | None:
        if self.config.scheduler_policy != "slo_aware":
            return self.scheduler.peek_next_waiting()
        waiting = self.scheduler.get_waiting_states()
        if not waiting:
            return None
        return max(waiting, key=self._waiting_priority_score)

    def _pop_waiting_state(self, state: RequestState) -> RequestState:
        if self.config.scheduler_policy != "slo_aware":
            return self.scheduler.pop_next_waiting()
        return self.scheduler.pop_waiting(state)

    def _remaining_decode_tokens(self, state: RequestState) -> int:
        return max(
            0,
            state.request.sampling_params.max_new_tokens - len(state.generated_token_ids),
        )

    def _reserved_blocks_for_swapped(self, state: RequestState) -> int:
        total_tokens = state.swapped_seq_len + self._remaining_decode_tokens(state)
        return max(1, math.ceil(total_tokens / self.config.block_size))

    def _projected_kv_utilization(self, blocks_needed: int) -> float:
        stats = self.kv_cache.stats()
        projected = int(stats["used_blocks"]) + blocks_needed
        return min(1.0, projected / self.config.num_gpu_blocks)

    def _adaptive_uses_reservation(self, blocks_needed: int) -> bool:
        return (
            self._projected_kv_utilization(blocks_needed)
            >= self.config.adaptive_reserve_threshold
            or self.scheduler.num_waiting() >= self.config.adaptive_waiting_threshold
        )

    def _adaptive_uses_pressure_preemption(self, blocks_needed: int) -> bool:
        return (
            self._projected_kv_utilization(blocks_needed)
            >= self.config.adaptive_preempt_threshold
            or self.scheduler.num_waiting() >= self.config.adaptive_waiting_threshold
        )

    def _uses_reservation(self, blocks_needed: int) -> bool:
        if self.config.scheduler_policy in {"reserve_only", "pressure_aware", "slo_aware"}:
            return True
        if self.config.scheduler_policy == "adaptive":
            return self._adaptive_uses_reservation(blocks_needed)
        return False

    def _uses_pressure_preemption(self, blocks_needed: int) -> bool:
        if self.config.scheduler_policy in {"pressure_aware", "slo_aware"}:
            return True
        if self.config.scheduler_policy == "adaptive":
            return self._adaptive_uses_pressure_preemption(blocks_needed)
        return False

    def _can_admit_blocks(self, blocks_needed: int) -> bool:
        if self._uses_reservation(blocks_needed):
            return self.kv_cache.can_reserve(
                blocks_needed,
                include_reclaimable=self._uses_pressure_preemption(blocks_needed),
            )
        return self.kv_cache.num_free_blocks() >= blocks_needed

    def _init_request_for_policy(self, state: RequestState, blocks_needed: int) -> None:
        if self._uses_reservation(blocks_needed):
            self.kv_cache.init_request(state, reserved_blocks=blocks_needed)
        else:
            self.kv_cache.init_request(state)
        state.admit_step = self._step_idx

    def _can_swap_in(self, state: RequestState) -> bool:
        reserved_blocks = self._reserved_blocks_for_swapped(state)
        if self._uses_reservation(reserved_blocks):
            return self.kv_cache.can_reserve(
                reserved_blocks,
                include_reclaimable=self._uses_pressure_preemption(reserved_blocks),
            )
        needed = max(1, math.ceil(state.swapped_seq_len / self.config.block_size))
        return self.kv_cache.num_free_blocks() >= needed

    def _swap_in_for_policy(self, state: RequestState) -> None:
        reserved_blocks = self._reserved_blocks_for_swapped(state)
        if self._uses_reservation(reserved_blocks):
            self.kv_cache.swap_in(
                state,
                reserved_blocks=reserved_blocks,
            )
        else:
            self.kv_cache.swap_in(state)

    def _select_victim(
        self,
        next_state: RequestState,
        blocks_needed: int,
    ) -> RequestState | None:
        if self._uses_pressure_preemption(blocks_needed):
            if self.config.scheduler_policy == "slo_aware":
                return self._select_slo_victim(next_state, blocks_needed)
            return self._select_pressure_victim(next_state, blocks_needed)

        victim = self.scheduler.get_lowest_priority_running()
        if victim is not None and victim.request.priority > next_state.request.priority:
            return victim
        return None

    def _select_pressure_victim(
        self,
        next_state: RequestState,
        blocks_needed: int,
    ) -> RequestState | None:
        """
        KV pressure-aware victim selection.

        优先抢占优先级更低的 running 请求；同优先级时，只抢占预留块更多的请求，
        用于让短请求绕过长请求，同时避免同尺寸请求互相抢占。
        """
        best_state: RequestState | None = None
        best_score: tuple[int, int, int, int, int] | None = None
        next_remaining = next_state.request.sampling_params.max_new_tokens

        for candidate in self.scheduler.get_running_states():
            rid = candidate.request.request_id
            candidate_reserved = self.kv_cache.get_reserved_blocks(rid)
            candidate_remaining = self._remaining_decode_tokens(candidate)
            candidate_generated = len(candidate.generated_token_ids)
            priority_gap = candidate.request.priority - next_state.request.priority

            if priority_gap > 0:
                # 数值越大优先级越低；优先牺牲低优先级、大 KV、剩余长的请求。
                score = (
                    2,
                    priority_gap,
                    candidate_reserved,
                    candidate_remaining,
                    -candidate_generated,
                )
            elif (
                priority_gap == 0
                and rid not in self._same_priority_preempted
                and candidate_reserved > blocks_needed
                and candidate_remaining >= next_remaining
            ):
                # 同优先级只做 size-aware preemption：短请求可以挤掉明显更大的长请求。
                score = (
                    1,
                    candidate_reserved - blocks_needed,
                    candidate_remaining - next_remaining,
                    candidate_reserved,
                    -candidate_generated,
                )
            else:
                continue

            if best_score is None or score > best_score:
                best_score = score
                best_state = candidate

        return best_state

    def _select_slo_victim(
        self,
        next_state: RequestState,
        blocks_needed: int,
    ) -> RequestState | None:
        """
        SLO-aware victim selection.

        Compared with pressure-aware, this protects requests that are close to
        finishing and allows an aged same-priority waiter to preempt a long
        running request once it becomes a TTFT SLO risk.
        """
        best_state: RequestState | None = None
        best_score: tuple[int, int, int, int, int, int, int] | None = None
        next_remaining = next_state.request.sampling_params.max_new_tokens
        next_slo_risk = self._request_slo_risk(next_state)

        for candidate in self.scheduler.get_running_states():
            rid = candidate.request.request_id
            candidate_reserved = self.kv_cache.get_reserved_blocks(rid)
            candidate_remaining = self._remaining_decode_tokens(candidate)
            candidate_generated = len(candidate.generated_token_ids)
            candidate_wait = self._request_wait_steps(candidate)
            priority_gap = candidate.request.priority - next_state.request.priority
            near_finish = candidate_remaining <= 2

            if candidate.request.priority < next_state.request.priority:
                # Do not sacrifice higher-priority running requests.
                continue

            if priority_gap > 0:
                category = 3
            elif (
                priority_gap == 0
                and rid not in self._same_priority_preempted
                and next_slo_risk > 0
                and not near_finish
                and candidate_reserved >= blocks_needed
                and candidate_remaining >= max(2, next_remaining // 2)
            ):
                # Same priority: only intervene once the waiter is at SLO risk.
                category = 2
            elif (
                priority_gap == 0
                and rid not in self._same_priority_preempted
                and candidate_reserved > blocks_needed
                and candidate_remaining >= next_remaining
            ):
                # Keep the previous short-over-long behavior.
                category = 1
            else:
                continue

            score = (
                category,
                priority_gap,
                next_slo_risk,
                candidate_reserved,
                candidate_remaining,
                -candidate_wait,
                -candidate_generated,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_state = candidate

        return best_state

    def _preempt_victim(
        self,
        victim: RequestState,
        newly_admitted: list[RequestState] | None = None,
        requester: RequestState | None = None,
    ) -> None:
        """执行 un_admit 或 swap_out，并维护 newly_admitted 列表。"""
        if requester is not None and victim.request.priority == requester.request.priority:
            self._same_priority_preempted.add(victim.request.request_id)
        if not victim.prefilled:
            self.kv_cache.free_request(victim)
            self.scheduler.un_admit(victim)
            if newly_admitted is not None and victim in newly_admitted:
                newly_admitted.remove(victim)
        else:
            self.kv_cache.swap_out(victim)
            self.scheduler.mark_swapped(victim)

    # ------------------------------------------------------------------
    # Phase 10：Prefix Cache 辅助方法（仅用于 chunk_prefill_size == 0 路径）
    # ------------------------------------------------------------------

    def _admit_with_prefix(self, state: RequestState, new_reserved_blocks: int | None = None) -> None:
        """准入时查找 prefix cache：命中则 init_request_with_prefix，否则 init_request。"""
        cached_len, cached_blocks = self.kv_cache.find_prefix_cache(state.prompt_token_ids)
        if cached_len > 0:
            if new_reserved_blocks is None:
                suffix_len = len(state.prompt_token_ids) - cached_len
                max_out = state.request.sampling_params.max_new_tokens
                new_reserved_blocks = math.ceil((suffix_len + max_out) / self.config.block_size)
            if self._uses_reservation(new_reserved_blocks):
                self.kv_cache.init_request_with_prefix(
                    state,
                    cached_len,
                    cached_blocks,
                    reserved_blocks=len(cached_blocks) + new_reserved_blocks,
                )
            else:
                self.kv_cache.init_request_with_prefix(state, cached_len, cached_blocks)
            state.prefix_cached_len = cached_len
            state.prefix_cached_blocks = cached_blocks
        else:
            if new_reserved_blocks is None:
                prompt_len = len(state.prompt_token_ids)
                max_out = state.request.sampling_params.max_new_tokens
                new_reserved_blocks = math.ceil((prompt_len + max_out) / self.config.block_size)
            self._init_request_for_policy(state, new_reserved_blocks)
        if state.admit_step is None:
            state.admit_step = self._step_idx

    def _prefill_and_register(self, newly_admitted: list[RequestState]) -> None:
        """对新准入请求执行 prefill，然后将结果注册到 prefix cache。

        prefix cache miss 的请求批量调用 prefill()；
        prefix cache hit 的请求逐个调用 prefill_with_prefix()。
        所有请求 prefill 完成后统一调用 register_prefix_blocks_for_request() 注册缓存。
        """
        miss_states = [s for s in newly_admitted if s.prefix_cached_len == 0]
        hit_states = [s for s in newly_admitted if s.prefix_cached_len > 0]
        if miss_states:
            self.model_runner.prefill(miss_states)
        for state in hit_states:
            self.model_runner.prefill_with_prefix(
                state, state.prefix_cached_len, state.prefix_cached_blocks
            )
        # 注册所有新 prefill 的 block（含 miss 和 hit 的 suffix blocks）
        for state in newly_admitted:
            self.kv_cache.register_prefix_blocks_for_request(
                state.request.request_id, state.prompt_token_ids
            )

    # ------------------------------------------------------------------
    # Phase 8：单步接口（供 AsyncEngine / HTTP 服务使用）
    # ------------------------------------------------------------------

    def add_request(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        priority: int = 0,
        request_id: str | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        """
        将单个请求加入等待队列，返回 request_id。

        request_id：外部预生成的 ID（供 AsyncEngine 在注册 token queue 后再调用此方法，
                    消除"queue 未注册时 step() 先投递 token"的竞态）。
                    若为 None，内部自动生成。

        与 step() 配合使用，实现流式/异步推理。
        不得与 generate() 同时混用——同一时刻只用一种接口。
        """
        if request_id is None:
            request_id = str(uuid4())
        request = Request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=SamplingParams(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            ),
            priority=priority,
        )
        state = RequestState(
            request=request,
            prompt_token_ids=self._tokenize(prompt),
            arrival_step=self._step_idx,
        )
        self.scheduler.add_request(state)
        self._step_states[request_id] = state
        self._trace_kv(
            "add_request",
            request_id=request_id,
            prompt_len=len(state.prompt_token_ids),
            max_new_tokens=max_new_tokens,
            priority=priority,
        )
        return request_id

    def cleanup_request(self, request_id: str) -> None:
        """
        从内部追踪表中删除已完成请求的 state。

        供 AsyncEngine 在确认 DONE 哨兵已投递后调用，防止 _step_states 无限增长。
        调用前提：is_finished(request_id) 已返回 True。
        """
        self._step_states.pop(request_id, None)

    def get_finish_reason(self, request_id: str) -> str | None:
        """返回请求的内部结束原因（如 length / eos）。"""
        state = self._step_states.get(request_id)
        return state.finish_reason if state is not None else None

    def cancel_request(self, request_id: str) -> bool:
        """
        主动取消请求，并回收其 KV / CPU swap 资源。

        返回值：
          - True: 成功找到并移除了该请求
          - False: 请求不存在或已清理
        """
        state = self._step_states.pop(request_id, None)
        if state is None:
            return False

        removed = self.scheduler.remove_request(request_id)
        state = removed if removed is not None else state

        if request_id in self.kv_cache._block_tables:
            self.kv_cache.free_request(state)
        state.cpu_kv = None
        # Phase 9：清除中间 prefill cache
        self._prefilling_caches.pop(request_id, None)
        self._trace_kv("cancel_request", request_id=request_id)
        return removed is not None

    def has_unfinished_requests(self) -> bool:
        """True 当且仅当有 waiting、prefilling、running 或 swapped 请求。"""
        return (
            self.scheduler.has_waiting()
            or self.scheduler.has_prefilling()
            or self.scheduler.num_running() > 0
            or self.scheduler.has_swapped()
        )

    def is_finished(self, request_id: str) -> bool:
        """True 当且仅当该请求已完成生成。"""
        state = self._step_states.get(request_id)
        return state is not None and state.finished

    def step(self) -> dict[str, list[str]]:
        """
        执行一次 prefill + decode 迭代。

        返回 {request_id: [new_token_texts]}：本步每个请求新生成的 token 文本列表。
        完成的请求在本步最后一次出现（含最后 token），随后 is_finished() 返回 True。
        """
        new_tokens: dict[str, list[str]] = {}
        self._step_idx += 1
        self._trace_kv("step_start")

        try:
            # just_prefilled：本步刚完成 prefill（整体 prefill 或最后一个 chunk）进入 running 的请求。
            # 供下方 pre_lens 逻辑将其初始 pre 置 0，使 prefill 采样的第一个 token 被捕获进入 new_tokens。
            just_prefilled: list[RequestState] = []

            if self.config.chunk_prefill_size > 0:
                # ── Chunked Prefill 路径（Phase 9）────────────────────────────
                # 1a. 若无正在进行的 prefill，从 waiting 准入一个请求到 PREFILLING
                if (
                    not self.scheduler.has_prefilling()
                    and self.scheduler.has_waiting()
                    and (self.scheduler.num_running() + self.scheduler.num_prefilling())
                        < self.config.max_batch_size
                ):
                    next_state = self._peek_next_waiting_state()
                    assert next_state is not None
                    prompt_len = len(next_state.prompt_token_ids)
                    max_out = next_state.request.sampling_params.max_new_tokens
                    blocks_needed = math.ceil((prompt_len + max_out) / self.config.block_size)
                    if blocks_needed > self.config.num_gpu_blocks:
                        self.scheduler.record_reject()
                        raise RuntimeError(
                            f"请求 {next_state.request.request_id!r} 需要 {blocks_needed} 个 KV 块，"
                            f"超过系统总块数 {self.config.num_gpu_blocks}。"
                        )
                    if self._can_admit_blocks(blocks_needed):
                        state = self._pop_waiting_state(next_state)
                        self._init_request_for_policy(state, blocks_needed)
                        self.scheduler.add_to_prefilling(state)
                        self._trace_kv(
                            "admit_prefilling",
                            request_id=state.request.request_id,
                            blocks_needed=blocks_needed,
                            prompt_len=prompt_len,
                            max_new_tokens=max_out,
                        )
                    else:
                        victim = self._select_victim(next_state, blocks_needed)
                        if victim is not None:
                            self._preempt_victim(victim, requester=next_state)
                            if self._can_admit_blocks(blocks_needed):
                                state = self._pop_waiting_state(next_state)
                                self._init_request_for_policy(state, blocks_needed)
                                self.scheduler.add_to_prefilling(state)
                                self._trace_kv(
                                    "admit_prefilling_after_preempt",
                                    request_id=state.request.request_id,
                                    blocks_needed=blocks_needed,
                                    prompt_len=prompt_len,
                                    max_new_tokens=max_out,
                                )

                # 2a. 推进当前 PREFILLING 请求的一个 chunk
                pf_state = self.scheduler.get_next_prefilling()
                if pf_state is not None:
                    rid = pf_state.request.request_id
                    t_start = pf_state.prefilled_tokens
                    t_end = min(
                        t_start + self.config.chunk_prefill_size,
                        len(pf_state.prompt_token_ids),
                    )
                    is_last = (t_end == len(pf_state.prompt_token_ids))
                    cache = self._prefilling_caches.get(rid)
                    new_cache = self.model_runner.prefill_chunk(
                        pf_state, t_start, t_end, cache, is_last
                    )
                    self._trace_kv(
                        "prefill_chunk",
                        request_id=rid,
                        token_start=t_start,
                        token_end=t_end,
                        is_last_chunk=is_last,
                    )
                    if is_last:
                        self._prefilling_caches.pop(rid, None)
                        self.scheduler.move_prefilling_to_running(pf_state)
                        just_prefilled.append(pf_state)
                        self._trace_kv("prefill_complete", request_id=rid)
                    else:
                        self._prefilling_caches[rid] = new_cache

            else:
                # ── 原始路径（Phase 8 行为，chunk_prefill_size == 0）──────────
                # Phase 10：此路径支持 prefix cache（chunked prefill 路径不走 prefix cache）
                # 1. 准入（与 generate() 相同逻辑）
                newly_admitted: list[RequestState] = []
                while (
                    self.scheduler.has_waiting()
                    and self.scheduler.num_running() < self.config.max_batch_size
                ):
                    next_state = self._peek_next_waiting_state()
                    assert next_state is not None

                    prompt_len = len(next_state.prompt_token_ids)
                    max_out = next_state.request.sampling_params.max_new_tokens
                    # Phase 10：预先查询 prefix cache，仅对 suffix + decode 部分计算所需新块数
                    cached_len_peek, _ = self.kv_cache.find_prefix_cache(
                        next_state.prompt_token_ids
                    )
                    suffix_len = prompt_len - cached_len_peek
                    blocks_needed = math.ceil((suffix_len + max_out) / self.config.block_size)

                    if blocks_needed > self.config.num_gpu_blocks:
                        self.scheduler.record_reject()
                        raise RuntimeError(
                            f"请求 {next_state.request.request_id!r} 需要 {blocks_needed} 个 KV 块，"
                            f"超过系统总块数 {self.config.num_gpu_blocks}。"
                        )

                    if self._can_admit_blocks(blocks_needed):
                        state = self._pop_waiting_state(next_state)
                        self._admit_with_prefix(state, blocks_needed)  # Phase 10：prefix cache 感知准入
                        self.scheduler.add_to_running(state)
                        newly_admitted.append(state)
                        self._trace_kv(
                            "admit_running",
                            request_id=state.request.request_id,
                            blocks_needed=blocks_needed,
                            prompt_len=prompt_len,
                            suffix_len=suffix_len,
                            max_new_tokens=max_out,
                            prefix_cached_len=cached_len_peek,
                        )
                    else:
                        victim = self._select_victim(next_state, blocks_needed)
                        if victim is not None:
                            self._preempt_victim(victim, newly_admitted, requester=next_state)
                            continue

                        if self.scheduler.num_running() == 0 and not newly_admitted:
                            self.scheduler.record_reject()
                            raise RuntimeError(
                                f"请求 {next_state.request.request_id!r} 需要 {blocks_needed} 个块，"
                                f"仅有 {self.kv_cache.num_free_blocks()} 个空闲块，无法换出。"
                            )
                        break

                # 2. Prefill（Phase 10：prefix hit 请求走 prefill_with_prefix，完成后注册缓存）
                if newly_admitted:
                    self._prefill_and_register(newly_admitted)
                    just_prefilled = newly_admitted
                    self._trace_kv(
                        "prefill_complete",
                        request_ids=[s.request.request_id for s in newly_admitted],
                    )

            # ── 3. Decode（收集前记录 pre_lens，捕获 prefill + decode 两阶段的新 token）
            running = self.scheduler.get_running_states()
            pre_lens: dict[str, int] = {
                s.request.request_id: len(s.generated_token_ids) for s in running
            }
            # 对刚完成 prefill 进入 running 的请求：pre 置 0，使 prefill 采样的第一个 token 被捕获
            for s in just_prefilled:
                pre_lens[s.request.request_id] = 0

            if running:
                self.model_runner.decode_batch(running)
                self._trace_kv(
                    "decode_batch",
                    request_ids=[s.request.request_id for s in running],
                    batch_size=len(running),
                )

            # ── 4. 收集新 token（清理前，确保 finished 请求的最后 token 也被捕获）
            #
            # 注意：real GPU 路径的 prefill/decode_batch 始终以 "" 存入 generated_text_parts
            # （逐 token decode 对多字节字符会返回空串），因此不能依赖 generated_text_parts。
            # 改用增量 tokenizer.decode：decode(all_ids[:curr]) - decode(all_ids[:pre])
            # 的文本差值，确保 real GPU 路径也能正确生成可读文本。
            # dry_run 路径同样走此逻辑（_StubTokenizer.decode 返回 " [1] [2]..." 格式正确）。
            for state in self.scheduler.get_running_states():
                rid = state.request.request_id
                pre = pre_lens.get(rid, 0)
                curr = len(state.generated_token_ids)
                if curr > pre:
                    tok_ids = state.generated_token_ids
                    old_text = (
                        self.model_runner.tokenizer.decode(
                            tok_ids[:pre], skip_special_tokens=True
                        )
                        if pre > 0 else ""
                    )
                    new_text = self.model_runner.tokenizer.decode(
                        tok_ids[:curr], skip_special_tokens=True
                    )
                    state.decoded_text = new_text
                    delta = new_text[len(old_text):]
                    if delta:
                        new_tokens[rid] = [delta]

            # ── 5. 清理完成的请求 ────────────────────────────────────────
            for state in list(self.scheduler.get_running_states()):
                if state.finished:
                    self.kv_cache.free_request(state)
                    self.scheduler.finish_request(state)
                    self._trace_kv(
                        "finish_request",
                        request_id=state.request.request_id,
                        finish_reason=state.finish_reason,
                        generated_tokens=len(state.generated_token_ids),
                    )

            # ── 6. Swap_in ───────────────────────────────────────────────
            # 注意：检查时同时计入 num_prefilling()，防止 PREFILLING 请求完成后
            # 导致 running + 1 超出 max_batch_size（chunked prefill 场景下的边界条件）
            for swapped_state in self.scheduler.get_swapped_states():
                if (self.scheduler.num_running() + self.scheduler.num_prefilling()
                        >= self.config.max_batch_size):
                    break
                needed = self._reserved_blocks_for_swapped(swapped_state)
                if self._can_swap_in(swapped_state):
                    self._swap_in_for_policy(swapped_state)
                    self.scheduler.move_swapped_to_running(swapped_state)
                    self._trace_kv(
                        "swap_in",
                        request_id=swapped_state.request.request_id,
                        restored_seq_len=swapped_state.swapped_seq_len,
                    )
                else:
                    break

        except Exception:
            self._trace_kv("step_exception")
            for state in self.scheduler.get_running_states():
                try:
                    self.kv_cache.free_request(state)
                except Exception as free_exc:
                    import sys
                    print(
                        f"warning: free_request failed for {state.request.request_id!r}: {free_exc}",
                        file=sys.stderr,
                    )
            # Phase 9：归还 PREFILLING 请求的 GPU 块
            for state in self.scheduler.get_prefilling_states():
                try:
                    self.kv_cache.free_request(state)
                except Exception as free_exc:
                    import sys
                    print(
                        f"warning: free_request failed for {state.request.request_id!r}: {free_exc}",
                        file=sys.stderr,
                    )
            self._prefilling_caches.clear()
            for state in self.scheduler.get_swapped_states():
                state.cpu_kv = None
            raise

        return new_tokens
