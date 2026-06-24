"""
请求调度器，支持 continuous batching（Phase 2）、preemption / 优先级调度（Phase 7）
和 chunked prefill（Phase 9）。

核心接口：
  waiting 队列管理：add_request, has_waiting, peek_next_waiting, pop_next_waiting
  running 队列管理：add_to_running, get_running_states, finish_request
  Preemption（Phase 7）：mark_swapped, has_swapped, move_swapped_to_running,
                          get_lowest_priority_running, un_admit
  Chunked Prefill（Phase 9）：add_to_prefilling, has_prefilling, num_prefilling,
                               get_prefilling_states, get_next_prefilling,
                               move_prefilling_to_running, remove_from_prefilling

get_next_batch() 为 Phase 1 遗留接口，仅供 test_scheduler.py 使用，新代码请勿调用。
"""

from collections import deque

from ..core.request import RequestState


class Scheduler:
    """提供等待队列到运行队列的批次调度能力，Phase 2 支持 continuous batching，Phase 7 支持抢占。"""

    def __init__(self, max_batch_size: int) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size 必须大于 0")
        self.max_batch_size = max_batch_size
        self._waiting: deque[RequestState] = deque()
        self._running: dict[str, RequestState] = {}
        self._swapped: deque[RequestState] = deque()  # Phase 7：已换出到 CPU 的请求
        self._prefilling: dict[str, RequestState] = {}  # Phase 9：chunked prefill 进行中
        self._counters: dict[str, int] = {
            "admit_count": 0,
            "reject_count": 0,
            "preempt_count": 0,
            "un_admit_count": 0,
            "swap_out_count": 0,
            "swap_in_count": 0,
            "finish_count": 0,
        }

    def add_request(self, state: RequestState) -> None:
        self._waiting.append(state)

    # ------------------------------------------------------------------
    # Continuous batching 接口（Phase 2 新增）
    # ------------------------------------------------------------------

    def has_waiting(self) -> bool:
        return bool(self._waiting)

    def peek_next_waiting(self) -> RequestState | None:
        """查看等待队列头部请求，不移除。"""
        return self._waiting[0] if self._waiting else None

    def pop_next_waiting(self) -> RequestState:
        """取出等待队列头部请求（不加入 running）。"""
        return self._waiting.popleft()

    def get_waiting_states(self) -> list[RequestState]:
        """返回 waiting 队列快照，供策略层做非 FIFO 选择。"""
        return list(self._waiting)

    def pop_waiting(self, state: RequestState) -> RequestState:
        """从 waiting 队列移除指定请求。"""
        self._waiting.remove(state)
        return state

    def add_to_running(self, state: RequestState) -> None:
        """将已确认可运行的请求加入 running dict。"""
        self._running[state.request.request_id] = state
        self._counters["admit_count"] += 1

    def get_running_states(self) -> list[RequestState]:
        return list(self._running.values())

    # ------------------------------------------------------------------
    # Phase 7：Preemption 接口
    # ------------------------------------------------------------------

    def mark_swapped(self, state: RequestState) -> None:
        """将请求从 running 移到 swapped 队列（swap_out 后调用）。"""
        self._running.pop(state.request.request_id, None)
        self._swapped.append(state)
        self._counters["preempt_count"] += 1
        self._counters["swap_out_count"] += 1

    def has_swapped(self) -> bool:
        return bool(self._swapped)

    def num_swapped(self) -> int:
        return len(self._swapped)

    def get_swapped_states(self) -> list[RequestState]:
        """按换出顺序（FIFO）返回所有换出请求。"""
        return list(self._swapped)

    def move_swapped_to_running(self, state: RequestState) -> None:
        """将请求从 swapped 移到 running（swap_in 后调用，跳过 prefill 直接参与 decode）。"""
        try:
            self._swapped.remove(state)
        except ValueError:
            pass
        self._running[state.request.request_id] = state
        self._counters["swap_in_count"] += 1

    def un_admit(self, state: RequestState) -> None:
        """
        撤销准入：将请求从 running 移回 waiting 队尾（不走 swap_out 路径）。

        仅用于刚准入但尚未 prefill 的请求（state.prefilled == False）。
        放到队尾而非队头，避免与高优先级请求的循环抢占。
        """
        self._running.pop(state.request.request_id, None)
        self._waiting.append(state)
        self._counters["preempt_count"] += 1
        self._counters["un_admit_count"] += 1

    def get_lowest_priority_running(self) -> RequestState | None:
        """
        返回 running 中优先级最低的请求（priority 数值最大）。
        若 running 为空返回 None。优先级相同时返回最早加入的请求（dict 插入顺序）。
        """
        if not self._running:
            return None
        return max(self._running.values(), key=lambda s: s.request.priority)

    # ------------------------------------------------------------------
    # Phase 9：Chunked Prefill 接口
    # ------------------------------------------------------------------

    def add_to_prefilling(self, state: RequestState) -> None:
        """将请求加入 PREFILLING 队列（KV 块已通过 init_request 分配）。"""
        self._prefilling[state.request.request_id] = state
        self._counters["admit_count"] += 1

    def has_prefilling(self) -> bool:
        return bool(self._prefilling)

    def num_prefilling(self) -> int:
        return len(self._prefilling)

    def get_prefilling_states(self) -> list[RequestState]:
        return list(self._prefilling.values())

    def get_next_prefilling(self) -> RequestState | None:
        """返回当前 PREFILLING 中的（唯一一个）请求，供 engine 推进 chunk。"""
        if self._prefilling:
            return next(iter(self._prefilling.values()))
        return None

    def move_prefilling_to_running(self, state: RequestState) -> None:
        """最后一个 chunk 完成后，将请求从 PREFILLING 移到 RUNNING。"""
        self._prefilling.pop(state.request.request_id, None)
        self._running[state.request.request_id] = state

    def remove_from_prefilling(self, state: RequestState) -> None:
        """将请求从 PREFILLING 移回 WAITING（被抢占时调用）。

        ⚠️  调用方（engine）还必须同步执行以下操作，否则会产生状态污染：
              1. state.prefilled_tokens = 0      — 重置 chunk 进度
              2. engine._prefilling_caches.pop(rid, None) — 丢弃中间 DynamicCache
              3. kv_cache.free_request(state)    — 释放已分配的 GPU 块
          当前 engine 的抢占路径（preemption）只针对 _running 中的请求，
          PREFILLING 请求不在 _running 里，因此该方法目前不会被 engine 调用。
          若将来需要支持 PREFILLING 抢占，请先在 engine 中实现以上三步再调用此方法。
        """
        self._prefilling.pop(state.request.request_id, None)
        state.prefilled_tokens = 0  # 重置 chunk 进度，防止重新准入时从错误位置继续
        self._waiting.append(state)

    # ------------------------------------------------------------------
    # 共用接口
    # ------------------------------------------------------------------

    def finish_request(self, state: RequestState) -> None:
        self._running.pop(state.request.request_id, None)
        self._counters["finish_count"] += 1

    def record_reject(self) -> None:
        """记录一次调度层拒绝/无法准入事件。"""
        self._counters["reject_count"] += 1

    def stats(self) -> dict[str, object]:
        """返回调度队列长度和累计调度事件计数。"""
        return {
            "waiting": self.num_waiting(),
            "prefilling": self.num_prefilling(),
            "running": self.num_running(),
            "swapped": self.num_swapped(),
            "counters": dict(self._counters),
        }

    def remove_request(self, request_id: str) -> RequestState | None:
        """
        从 waiting / prefilling / running / swapped 中移除指定请求。

        供 AsyncEngine 超时、断连或错误恢复时主动取消请求使用。
        返回被移除的 RequestState；若不存在则返回 None。
        """
        state = self._running.pop(request_id, None)
        if state is not None:
            return state

        # Phase 9：也检查 prefilling 队列
        state = self._prefilling.pop(request_id, None)
        if state is not None:
            return state

        for queue in (self._waiting, self._swapped):
            for state in queue:
                if state.request.request_id == request_id:
                    queue.remove(state)
                    return state
        return None

    def num_waiting(self) -> int:
        return len(self._waiting)

    def num_running(self) -> int:
        return len(self._running)

    # ------------------------------------------------------------------
    # Phase 1 兼容接口（保留供测试使用）
    # ------------------------------------------------------------------

    def get_next_batch(self) -> list[RequestState]:
        """Phase 1 接口：一次取出最多 max_batch_size 个等待请求并加入 running。"""
        batch: list[RequestState] = []
        while self._waiting and len(batch) < self.max_batch_size:
            state = self._waiting.popleft()
            self._running[state.request.request_id] = state
            batch.append(state)
        return batch
