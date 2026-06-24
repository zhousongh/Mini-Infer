from __future__ import annotations

import json

from benchmarks.benchmark_http_streaming import parse_sse_data, percentile


def test_parse_sse_data_extracts_content() -> None:
    payload = {
        "choices": [
            {
                "delta": {"content": "hello"},
                "finish_reason": None,
            }
        ]
    }

    kind, content, finish_reason = parse_sse_data(json.dumps(payload))

    assert kind == "content"
    assert content == "hello"
    assert finish_reason is None


def test_parse_sse_data_extracts_finish_reason() -> None:
    payload = {
        "choices": [
            {
                "delta": {},
                "finish_reason": "length",
            }
        ]
    }

    kind, content, finish_reason = parse_sse_data(json.dumps(payload))

    assert kind == "finish"
    assert content is None
    assert finish_reason == "length"


def test_parse_sse_data_handles_done() -> None:
    assert parse_sse_data("[DONE]") == ("done", None, None)


def test_percentile_uses_nearest_rank_floor() -> None:
    assert percentile([10, 20, 30, 40], 0.50) == 20
    assert percentile([10, 20, 30, 40], 0.95) == 30
    assert percentile([], 0.95) == 0.0
