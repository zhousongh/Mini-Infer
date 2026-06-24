"""examples/openai_client.py — 通过 OpenAI 兼容接口调用 mini-infer。

只依赖 Python 标准库，无需安装 openai SDK。

使用方法
--------
步骤 1：启动服务（无需模型权重，dry-run 模式）

    python serve.py --dry-run --port 8000

或者使用真实模型（需要 Qwen2.5 系列）：

    python serve.py --model /path/to/Qwen2.5-1.5B-Instruct --port 8000

步骤 2：运行本示例

    python examples/openai_client.py

可选：连接到不同端口或地址

    BASE_URL=http://localhost:9000/v1 python examples/openai_client.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/v1")
MODEL = "mini-infer"


# ---------------------------------------------------------------------------
# API 调用函数
# ---------------------------------------------------------------------------


def list_models() -> list[str]:
    """GET /v1/models — 列出可用模型。"""
    req = urllib.request.Request(f"{BASE_URL}/models")
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    return [m["id"] for m in data.get("data", [])]


def chat(messages: list[dict], stream: bool = False, max_tokens: int = 128) -> str:
    """POST /v1/chat/completions — 支持 streaming 和 non-streaming。

    Parameters
    ----------
    messages:
        OpenAI 格式的消息列表，例如：
        [{"role": "user", "content": "..."}]
    stream:
        True 时使用 SSE 流式输出并实时打印；False 时等待完整响应。
    max_tokens:
        最大生成 token 数。

    Returns
    -------
    str
        模型生成的完整文本。
    """
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    if not stream:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    # SSE streaming
    full_text = ""
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload_str = line[len("data:"):].strip()
            if payload_str == "[DONE]":
                break
            chunk = json.loads(payload_str)
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                print(delta, end="", flush=True)
                full_text += delta
    print()  # 换行
    return full_text


# ---------------------------------------------------------------------------
# 示例主程序
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"连接到：{BASE_URL}\n")

    # 1. 列出模型
    try:
        models = list_models()
    except urllib.error.URLError as e:
        print(f"[错误] 无法连接到服务：{e}")
        print("请先启动服务：python serve.py --dry-run --port 8000")
        return 1

    print(f"可用模型：{models}\n")

    # 2. Non-streaming 示例
    print("=" * 60)
    print("Non-streaming 示例")
    print("=" * 60)
    messages = [
        {"role": "system", "content": "你是一个简洁的技术助手。"},
        {"role": "user", "content": "用一句话解释 PagedAttention 的核心思想。"},
    ]
    reply = chat(messages, stream=False)
    print(f"回复：{reply}\n")

    # 3. Streaming 示例
    print("=" * 60)
    print("Streaming 示例（实时输出）")
    print("=" * 60)
    messages2 = [
        {"role": "system", "content": "你是一个简洁的技术助手。"},
        {"role": "user", "content": "用一句话解释 Continuous Batching 的收益。"},
    ]
    print("回复：", end="")
    chat(messages2, stream=True)

    # 4. 多轮对话示例
    print("=" * 60)
    print("多轮对话示例")
    print("=" * 60)
    history: list[dict] = [
        {"role": "system", "content": "你是一个简洁的技术助手。"},
    ]
    turns = [
        "mini-infer 支持哪些并行策略？",
        "其中 Tensor Parallelism 和 Expert Parallelism 有什么区别？",
    ]
    for user_msg in turns:
        print(f"User: {user_msg}")
        history.append({"role": "user", "content": user_msg})
        reply = chat(history, stream=False)
        print(f"Assistant: {reply}\n")
        history.append({"role": "assistant", "content": reply})

    return 0


if __name__ == "__main__":
    sys.exit(main())
