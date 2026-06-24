from __future__ import annotations

import json

from benchmarks.analyze_kv_trace import analyze


def _record(step: int, event: str, **extra: object) -> dict[str, object]:
    record: dict[str, object] = {
        "ts": 100.0 + step * 0.01,
        "step": step,
        "event": event,
        "scheduler": {
            "waiting": 0,
            "prefilling": 0,
            "running": 1,
            "swapped": 0,
            "counters": {
                "admit_count": 1,
                "reject_count": 0,
                "preempt_count": 0,
                "un_admit_count": 0,
                "swap_out_count": 0,
                "swap_in_count": 0,
                "finish_count": 0,
            },
        },
        "kv_cache": {
            "total_blocks": 4,
            "free_blocks": 2,
            "used_blocks": 2,
            "reserved_blocks": 3,
            "utilization": 0.5,
        },
    }
    record.update(extra)
    return record


def test_analyze_kv_trace_recovers_request_lifecycle(tmp_path) -> None:
    trace_file = tmp_path / "kv_trace.jsonl"
    records = [
        _record(
            0,
            "add_request",
            request_id="req-1",
            prompt_len=8,
            max_new_tokens=3,
            priority=1,
        ),
        _record(1, "step_start"),
        _record(1, "admit_running", request_id="req-1", blocks_needed=3),
        _record(1, "prefill_complete", request_ids=["req-1"]),
        _record(1, "decode_batch", request_ids=["req-1"], batch_size=1),
        _record(
            3,
            "finish_request",
            request_id="req-1",
            finish_reason="length",
            generated_tokens=3,
            scheduler={
                "waiting": 0,
                "prefilling": 0,
                "running": 0,
                "swapped": 0,
                "counters": {
                    "admit_count": 1,
                    "reject_count": 0,
                    "preempt_count": 0,
                    "un_admit_count": 0,
                    "swap_out_count": 0,
                    "swap_in_count": 0,
                    "finish_count": 1,
                },
            },
        ),
    ]
    trace_file.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )

    analysis = analyze(trace_file)

    assert analysis.records == len(records)
    assert analysis.completed == 1
    assert analysis.peak_kv_utilization == 0.5
    assert analysis.peak_used_blocks == 2
    assert analysis.peak_reserved_blocks == 3
    assert analysis.scheduler_counters["finish_count"] == 1

    req = analysis.requests[0]
    assert req.request_id == "req-1"
    assert req.status == "finished"
    assert req.prompt_len == 8
    assert req.max_new_tokens == 3
    assert req.priority == 1
    assert req.add_step == 0
    assert req.admit_step == 1
    assert req.first_decode_step == 1
    assert req.finish_step == 3
    assert req.ttft_steps == 2
    assert req.latency_steps == 4
    assert req.generated_tokens == 3
