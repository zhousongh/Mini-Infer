"""这个文件覆盖调度器的批次行为和 Phase 2 continuous batching 接口，验证等待/运行队列的状态流转。"""

from mini_infer.core.request import Request, RequestState, SamplingParams
from mini_infer.runtime.scheduler import Scheduler


def build_state(request_id: str) -> RequestState:
    return RequestState(
        request=Request(
            request_id=request_id,
            prompt=f"prompt-{request_id}",
            sampling_params=SamplingParams(max_new_tokens=2),
        )
    )


def test_scheduler_batching() -> None:
    scheduler = Scheduler(max_batch_size=2)
    scheduler.add_request(build_state("a"))
    scheduler.add_request(build_state("b"))
    scheduler.add_request(build_state("c"))

    batch = scheduler.get_next_batch()

    assert [state.request.request_id for state in batch] == ["a", "b"]
    assert scheduler.num_waiting() == 1
    assert scheduler.num_running() == 2


# ── Phase 2 continuous batching 接口 ──────────────────────────────────────


def test_has_waiting_empty() -> None:
    scheduler = Scheduler(max_batch_size=4)
    assert not scheduler.has_waiting()


def test_has_waiting_nonempty() -> None:
    scheduler = Scheduler(max_batch_size=4)
    scheduler.add_request(build_state("x"))
    assert scheduler.has_waiting()


def test_peek_next_waiting_returns_none_when_empty() -> None:
    scheduler = Scheduler(max_batch_size=4)
    assert scheduler.peek_next_waiting() is None


def test_peek_does_not_remove() -> None:
    scheduler = Scheduler(max_batch_size=4)
    scheduler.add_request(build_state("a"))
    scheduler.add_request(build_state("b"))

    peeked = scheduler.peek_next_waiting()
    assert peeked is not None
    assert peeked.request.request_id == "a"
    # 调用两次 peek，队列不变
    peeked2 = scheduler.peek_next_waiting()
    assert peeked2 is not None
    assert peeked2.request.request_id == "a"
    assert scheduler.num_waiting() == 2


def test_pop_next_waiting_removes_head() -> None:
    scheduler = Scheduler(max_batch_size=4)
    scheduler.add_request(build_state("a"))
    scheduler.add_request(build_state("b"))

    popped = scheduler.pop_next_waiting()
    assert popped.request.request_id == "a"
    assert scheduler.num_waiting() == 1
    assert scheduler.peek_next_waiting().request.request_id == "b"  # type: ignore[union-attr]


def test_add_to_running_and_get_running_states() -> None:
    scheduler = Scheduler(max_batch_size=4)
    state_a = build_state("a")
    state_b = build_state("b")

    scheduler.add_request(state_a)
    scheduler.add_request(state_b)

    # 手动取出并加入 running（模拟 continuous batching 主循环）
    scheduler.add_to_running(scheduler.pop_next_waiting())
    scheduler.add_to_running(scheduler.pop_next_waiting())

    running = scheduler.get_running_states()
    assert len(running) == 2
    running_ids = {s.request.request_id for s in running}
    assert running_ids == {"a", "b"}
    assert scheduler.num_waiting() == 0


def test_finish_request_removes_from_running() -> None:
    scheduler = Scheduler(max_batch_size=4)
    state = build_state("a")
    scheduler.add_request(state)
    scheduler.add_to_running(scheduler.pop_next_waiting())

    assert scheduler.num_running() == 1
    scheduler.finish_request(state)
    assert scheduler.num_running() == 0
    assert scheduler.get_running_states() == []


def test_continuous_batching_admit_loop() -> None:
    """模拟 engine.py 中的准入循环：当 running < max_batch_size 时逐个准入等待请求。"""
    scheduler = Scheduler(max_batch_size=2)
    for rid in ["a", "b", "c", "d"]:
        scheduler.add_request(build_state(rid))

    # 第一轮准入：填满 running
    while scheduler.has_waiting() and scheduler.num_running() < scheduler.max_batch_size:
        scheduler.add_to_running(scheduler.pop_next_waiting())

    assert scheduler.num_running() == 2
    assert scheduler.num_waiting() == 2

    # 完成一个请求
    finished = scheduler.get_running_states()[0]
    scheduler.finish_request(finished)
    assert scheduler.num_running() == 1

    # 准入下一个
    if scheduler.has_waiting() and scheduler.num_running() < scheduler.max_batch_size:
        scheduler.add_to_running(scheduler.pop_next_waiting())

    assert scheduler.num_running() == 2
    assert scheduler.num_waiting() == 1
