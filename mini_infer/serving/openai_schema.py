"""
Phase 8 OpenAI Chat Completions 子集兼容 schema。

定义 mini-infer 当前支持的 Chat Completions 受限子集请求/响应 Pydantic 模型，
供 server.py 的路由层使用。

覆盖接口：
  POST /v1/chat/completions（streaming + non-streaming）
  GET  /v1/models
"""

from __future__ import annotations

import time
from typing import List, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: List[ChatMessage]
    stream: bool = False
    max_tokens: int = Field(default=128, ge=1)
    temperature: float = Field(default=0.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    # 以下字段保留在 schema 中用于基础 SDK 兼容；当前服务仅支持默认值
    n: int = Field(default=1, ge=1, le=1)
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: list[str] | str | None = None


# ---------------------------------------------------------------------------
# 非流式响应模型
# ---------------------------------------------------------------------------


class ChatCompletionMessage(BaseModel):
    role: str = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionMessage
    finish_reason: str | None = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# ---------------------------------------------------------------------------
# 流式响应模型（SSE chunk）
# ---------------------------------------------------------------------------


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatCompletionChunkChoice]


# ---------------------------------------------------------------------------
# /v1/models 端点模型
# ---------------------------------------------------------------------------


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "local"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]
