"""
Phase 8/9 OpenAI Chat Completions 子集兼容 HTTP server。

提供 mini-infer 当前支持的 Chat Completions 受限子集接口：
  GET  /v1/models
  POST /v1/chat/completions（streaming + non-streaming）

使用方式：
  python serve.py --model /path/to/model

或直接通过 uvicorn：
  # 未注入 app.state.engine_config 时，默认回退到 dry_run 配置
  uvicorn mini_infer.serving.server:app --host 0.0.0.0 --port 8000
  # 如需真实模型，可先设置环境变量
  MINI_INFER_MODEL=/path/to/model uvicorn mini_infer.serving.server:app --host 0.0.0.0 --port 8000
  # 开启 chunked prefill
  MINI_INFER_MODEL=/path/to/model MINI_INFER_CHUNK_PREFILL_SIZE=256 uvicorn mini_infer.serving.server:app --host 0.0.0.0 --port 8000
  # 开启 CUDA Graph + W8A8 量化
  MINI_INFER_MODEL=/path/to/model MINI_INFER_USE_CUDA_GRAPH=1 MINI_INFER_QUANT_MODE=w8a8 uvicorn mini_infer.serving.server:app --host 0.0.0.0 --port 8000

启动时全局初始化 AsyncEngine，所有请求共享同一 step loop，
实现 continuous batching（多并发 HTTP 请求被合并进同一 decode_batch）。
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..core.config import EngineConfig
from ..runtime.async_engine import AsyncEngine
from .openai_schema import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    DeltaMessage,
    ModelCard,
    ModelList,
    Usage,
)

# ---------------------------------------------------------------------------
# 全局 engine 实例（由 lifespan 初始化）
# ---------------------------------------------------------------------------

_engine: AsyncEngine | None = None
_model_id: str = "mini-infer"


def _parse_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _default_engine_config() -> EngineConfig:
    model_name = os.getenv("MINI_INFER_MODEL", "dry")
    dry_run = _parse_env_bool("MINI_INFER_DRY_RUN", model_name == "dry")
    return EngineConfig(
        model_name=model_name,
        device=os.getenv("MINI_INFER_DEVICE", "cuda:0"),
        dtype=os.getenv("MINI_INFER_DTYPE", "float16"),
        dry_run=dry_run,
        max_batch_size=int(os.getenv("MINI_INFER_MAX_BATCH_SIZE", "8")),
        num_gpu_blocks=int(os.getenv("MINI_INFER_NUM_GPU_BLOCKS", "200")),
        block_size=int(os.getenv("MINI_INFER_BLOCK_SIZE", "256")),
        chunk_prefill_size=int(os.getenv("MINI_INFER_CHUNK_PREFILL_SIZE", "0")),
        use_cuda_graph=_parse_env_bool("MINI_INFER_USE_CUDA_GRAPH", False),
        quant_mode=os.getenv("MINI_INFER_QUANT_MODE", ""),
        scheduler_policy=os.getenv("MINI_INFER_SCHEDULER_POLICY", "pressure_aware"),
        adaptive_reserve_threshold=float(os.getenv("MINI_INFER_ADAPTIVE_RESERVE_THRESHOLD", "0.60")),
        adaptive_preempt_threshold=float(os.getenv("MINI_INFER_ADAPTIVE_PREEMPT_THRESHOLD", "0.85")),
        adaptive_waiting_threshold=int(os.getenv("MINI_INFER_ADAPTIVE_WAITING_THRESHOLD", "8")),
        scheduler_ttft_slo_steps=int(os.getenv("MINI_INFER_SCHEDULER_TTFT_SLO_STEPS", "16")),
        scheduler_latency_slo_steps=int(os.getenv("MINI_INFER_SCHEDULER_LATENCY_SLO_STEPS", "64")),
    )


def _model_to_json(model: BaseModel) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(exclude_none=True)
    return model.json(exclude_none=True)


def _normalize_finish_reason(reason: str | None) -> str:
    if reason in {None, "eos", "stop"}:
        return "stop"
    if reason == "length":
        return "length"
    return "stop"


def _validate_request(request: ChatCompletionRequest) -> None:
    if request.model != _model_id:
        raise HTTPException(
            status_code=404,
            detail=f"Model {request.model!r} not found. Available model: {_model_id!r}.",
        )
    unsupported: list[str] = []
    if request.stop not in (None, "", []):
        unsupported.append("stop")
    if request.presence_penalty != 0.0:
        unsupported.append("presence_penalty")
    if request.frequency_penalty != 0.0:
        unsupported.append("frequency_penalty")
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported request fields: {', '.join(unsupported)}",
        )


def get_engine() -> AsyncEngine:
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    try:
        _engine.ensure_healthy()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _engine


# ---------------------------------------------------------------------------
# Lifespan：启动/关闭 AsyncEngine
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _engine
    config = getattr(app.state, "engine_config", None)
    if config is None:
        config = _default_engine_config()
        app.state.engine_config = config  # type: ignore[attr-defined]
    _engine = AsyncEngine(config)
    await _engine.start()
    yield
    await _engine.stop()
    _engine = None


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="mini-infer", lifespan=lifespan)


# ---------------------------------------------------------------------------
# GET /healthz
# ---------------------------------------------------------------------------


@app.get("/healthz")
async def healthz() -> dict:
    """健康检查端点，返回引擎状态。"""
    if _engine is None:
        return {"status": "initializing"}
    try:
        _engine.ensure_healthy()
        config = app.state.engine_config
        return {
            "status": "ok",
            "model": config.model_name,
            "device": config.device,
            "use_cuda_graph": config.use_cuda_graph,
            "quant_mode": config.quant_mode or "none",
            "chunk_prefill_size": config.chunk_prefill_size,
            "scheduler_policy": config.scheduler_policy,
            "adaptive_reserve_threshold": config.adaptive_reserve_threshold,
            "adaptive_preempt_threshold": config.adaptive_preempt_threshold,
            "adaptive_waiting_threshold": config.adaptive_waiting_threshold,
            "scheduler_ttft_slo_steps": config.scheduler_ttft_slo_steps,
            "scheduler_latency_slo_steps": config.scheduler_latency_slo_steps,
            "kv_cache": _engine.kv_cache_stats(),
            "scheduler": _engine.scheduler_stats(),
        }
    except RuntimeError as exc:
        return {"status": "error", "detail": str(exc)}


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------


@app.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    return ModelList(data=[ModelCard(id=_model_id, created=0)])


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):  # type: ignore[return]
    engine = get_engine()
    _validate_request(request)
    prompt = engine.format_prompt(request.messages)

    if request.stream:
        return StreamingResponse(
            _stream_generator(engine, prompt, request),
            media_type="text/event-stream",
        )
    else:
        return await _non_stream(engine, prompt, request)


# ---------------------------------------------------------------------------
# Non-streaming response
# ---------------------------------------------------------------------------


async def _non_stream(
    engine: AsyncEngine,
    prompt: str,
    request: ChatCompletionRequest,
) -> ChatCompletionResponse:
    text, finish_reason = await engine.generate_with_reason(
        prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    )
    completion_id = f"chatcmpl-{uuid4().hex[:8]}"
    tokenizer = engine.tokenizer
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    completion_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    return ChatCompletionResponse(
        id=completion_id,
        model=_model_id,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatCompletionMessage(role="assistant", content=text),
                finish_reason=_normalize_finish_reason(finish_reason),
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


# ---------------------------------------------------------------------------
# Streaming response（SSE）
# ---------------------------------------------------------------------------


async def _stream_generator(
    engine: AsyncEngine,
    prompt: str,
    request: ChatCompletionRequest,
) -> AsyncGenerator[str, None]:
    completion_id = f"chatcmpl-{uuid4().hex[:8]}"
    created = int(time.time())
    finish_reason = "stop"

    # 第一个 chunk：role only
    first_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=_model_id,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=DeltaMessage(role="assistant"),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {_model_to_json(first_chunk)}\n\n"

    # 逐增量文本 chunk（一次 step 可能合并多个 token 到同一个 delta）
    async for event in engine.generate_stream_events(
        prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
    ):
        if hasattr(event, "text"):
            chunk = ChatCompletionChunk(
                id=completion_id,
                created=created,
                model=_model_id,
                choices=[
                    ChatCompletionChunkChoice(
                        index=0,
                        delta=DeltaMessage(content=event.text),
                        finish_reason=None,
                    )
                ],
            )
            yield f"data: {_model_to_json(chunk)}\n\n"
        else:
            finish_reason = _normalize_finish_reason(event.finish_reason)

    # 结束 chunk
    stop_chunk = ChatCompletionChunk(
        id=completion_id,
        created=created,
        model=_model_id,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=DeltaMessage(),
                finish_reason=finish_reason,
            )
        ],
    )
    yield f"data: {_model_to_json(stop_chunk)}\n\n"
    yield "data: [DONE]\n\n"
