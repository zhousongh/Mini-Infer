"""
Phase 8 HTTP server 测试。

覆盖范围：
  - GET  /v1/models
  - POST /v1/chat/completions（non-streaming）
  - POST /v1/chat/completions（streaming SSE）

使用 httpx.AsyncClient + ASGITransport 直接调用 ASGI app（无需启动真实服务器）。
引擎使用 dry_run=True，无需真实模型权重。
"""

from __future__ import annotations

import json
import sys

import pytest
import pytest_asyncio
import httpx
from asgi_lifespan import LifespanManager

from mini_infer.core.config import EngineConfig
from mini_infer.serving.server import app, _default_engine_config
import serve


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client():
    """启动 dry_run AsyncEngine，通过 ASGI transport 直接访问 app。"""
    config = EngineConfig(
        model_name="dry",
        dry_run=True,
        num_gpu_blocks=32,
        block_size=256,
        max_batch_size=4,
    )
    app.state.engine_config = config  # type: ignore[attr-defined]
    async with LifespanManager(app) as manager:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=manager.app), base_url="http://test"
        ) as c:
            yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_models(client: httpx.AsyncClient):
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) >= 1
    assert "id" in body["data"][0]


@pytest.mark.asyncio
async def test_healthz_exposes_kv_cache_stats(client: httpx.AsyncClient):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["kv_cache"]["total_blocks"] == 32
    assert body["kv_cache"]["free_blocks"] == 32
    assert body["kv_cache"]["used_blocks"] == 0
    assert body["kv_cache"]["utilization"] == 0.0
    assert body["kv_cache"]["active_requests"] == 0
    assert body["scheduler"]["waiting"] == 0
    assert body["scheduler"]["running"] == 0
    assert body["scheduler"]["swapped"] == 0
    assert body["scheduler"]["counters"]["admit_count"] == 0
    assert body["scheduler"]["counters"]["reject_count"] == 0


@pytest.mark.asyncio
async def test_chat_completion_non_stream(client: httpx.AsyncClient):
    payload = {
        "model": "mini-infer",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 3,
        "stream": False,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "chat.completion"
    assert len(body["choices"]) == 1
    choice = body["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert isinstance(choice["message"]["content"], str)
    assert len(choice["message"]["content"]) > 0
    assert choice["finish_reason"] == "length"


@pytest.mark.asyncio
async def test_chat_completion_stream(client: httpx.AsyncClient):
    payload = {
        "model": "mini-infer",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 3,
        "stream": True,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    lines = resp.text.strip().split("\n")
    data_lines = [l[6:] for l in lines if l.startswith("data: ")]

    # 最后一行是 [DONE]
    assert data_lines[-1] == "[DONE]"

    # 解析所有 JSON chunk（排除 [DONE]）
    chunks = [json.loads(d) for d in data_lines[:-1]]
    assert len(chunks) >= 3  # 至少 role chunk + 1 token chunk + stop chunk

    # 第一个 chunk 包含 role
    first = chunks[0]
    assert first["object"] == "chat.completion.chunk"
    assert first["choices"][0]["delta"].get("role") == "assistant"

    # 最后一个 JSON chunk 的 finish_reason == "length"
    last_chunk = chunks[-1]
    assert last_chunk["choices"][0]["finish_reason"] == "length"

    # 中间 chunk 有 content
    content_chunks = [c for c in chunks[1:-1] if c["choices"][0]["delta"].get("content")]
    assert len(content_chunks) > 0


@pytest.mark.asyncio
async def test_stream_has_valid_ids(client: httpx.AsyncClient):
    """同一次请求的所有 chunk 应共享同一个 completion id。"""
    payload = {
        "model": "mini-infer",
        "messages": [{"role": "user", "content": "test"}],
        "max_tokens": 3,
        "stream": True,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    data_lines = [l[6:] for l in resp.text.strip().split("\n") if l.startswith("data: ")]
    chunks = [json.loads(d) for d in data_lines if d != "[DONE]"]
    ids = {c["id"] for c in chunks}
    assert len(ids) == 1  # 所有 chunk 共享同一 id


@pytest.mark.asyncio
async def test_invalid_max_tokens_returns_422(client: httpx.AsyncClient):
    payload = {
        "model": "mini-infer",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 0,
        "stream": False,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_unknown_model_returns_404(client: httpx.AsyncClient):
    payload = {
        "model": "not-a-real-model",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 2,
        "stream": False,
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_unsupported_stop_returns_400(client: httpx.AsyncClient):
    payload = {
        "model": "mini-infer",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 2,
        "stream": False,
        "stop": ["\n\n"],
    }
    resp = await client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_server_can_start_without_injected_config():
    """不经过 serve.py 注入 config 时，server.py 默认回退到 dry_run 配置。"""
    had_config = hasattr(app.state, "engine_config")
    old_config = getattr(app.state, "engine_config", None)
    if had_config:
        delattr(app.state, "engine_config")
    try:
        async with LifespanManager(app) as manager:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=manager.app), base_url="http://test"
            ) as c:
                resp = await c.get("/v1/models")
                assert resp.status_code == 200
                body = resp.json()
                assert body["data"][0]["id"] == "mini-infer"
    finally:
        if had_config:
            app.state.engine_config = old_config  # type: ignore[attr-defined]


def test_default_engine_config_reads_chunk_prefill_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINI_INFER_MODEL", "dry")
    monkeypatch.setenv("MINI_INFER_CHUNK_PREFILL_SIZE", "256")
    config = _default_engine_config()
    assert config.chunk_prefill_size == 256


def test_serve_parse_args_exposes_chunk_prefill_size(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["serve.py", "--dry-run", "--chunk-prefill-size", "256"],
    )
    args = serve.parse_args()
    assert args.chunk_prefill_size == 256
