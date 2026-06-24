"""examples/local_chat.py — 不启动 HTTP 服务，直接使用 Python API 与 mini-infer 交互。

与 examples/openai_client.py 的区别
-----------------------------------
- local_chat.py  ：直接调用 LLMEngine Python API，无需 HTTP 服务，无网络延迟
- openai_client.py：通过 OpenAI 兼容接口，需要先启动 mini-infer-serve

使用方法
--------
方式 A — dry-run（无需模型权重，验证安装是否正确）

    python examples/local_chat.py --dry-run

方式 B — 真实模型（流式 step 循环，展示 add_request / step / is_finished 接口）

    python examples/local_chat.py --model /path/to/Qwen2.5-1.5B-Instruct

方式 C — 批量生成（generate 接口，最简洁）

    python examples/local_chat.py --model /path/to/Qwen2.5-1.5B-Instruct --batch
"""

from __future__ import annotations

import argparse
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mini-infer local chat example")
    parser.add_argument("--model", type=str, default="", help="模型目录路径")
    parser.add_argument("--dry-run", action="store_true", help="stub model，无需真实权重")
    parser.add_argument("--batch", action="store_true", help="使用 generate() 而非 step 循环")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0, help="0.0 = greedy")
    parser.add_argument("--block-size", type=int, default=256)
    return parser.parse_args()


def build_engine(args: argparse.Namespace):
    from mini_infer.core.config import EngineConfig
    from mini_infer.runtime.engine import LLMEngine

    if not args.dry_run and not args.model:
        raise SystemExit("请指定 --model 或 --dry-run")

    config = EngineConfig(
        model_name=args.model if args.model else "dry",
        device=args.device,
        dtype=args.dtype,
        dry_run=args.dry_run,
        block_size=args.block_size,
    )
    print(f"[local_chat] model={config.model_name!r}  dry_run={config.dry_run}")
    return LLMEngine(config)


# ── 接口演示 A：add_request + step + is_finished（逐 token 流式）──────────────

def generate_streaming(engine, prompt: str, max_tokens: int, temperature: float) -> tuple[str, float]:
    """
    使用 add_request / step / is_finished 接口。

    engine.add_request(prompt, max_new_tokens, temperature) -> request_id
    engine.step()  -> {request_id: [new_token_texts]}
    engine.is_finished(request_id) -> bool
    """
    t0 = time.perf_counter()
    request_id = engine.add_request(prompt, max_new_tokens=max_tokens, temperature=temperature)

    parts: list[str] = []
    while not engine.is_finished(request_id):
        new_tokens: dict[str, list[str]] = engine.step()
        parts.extend(new_tokens.get(request_id, []))

    return "".join(parts), time.perf_counter() - t0


# ── 接口演示 B：generate()（批量，最简洁）────────────────────────────────────

def generate_batch(engine, prompt: str, max_tokens: int, temperature: float) -> tuple[str, float]:
    """使用 engine.generate([prompt], ...) 批量接口，一次性拿到完整结果。"""
    t0 = time.perf_counter()
    results = engine.generate([prompt], max_new_tokens=max_tokens, temperature=temperature)
    return results[0], time.perf_counter() - t0


# ── 交互循环 ─────────────────────────────────────────────────────────────────

def chat_loop(engine, args: argparse.Namespace) -> int:
    generate_fn = generate_batch if args.batch else generate_streaming
    mode = "generate()" if args.batch else "add_request/step/is_finished"

    print(f"\n=== mini-infer local chat  [{mode}] ===")
    print("输入 'quit' 退出，'clear' 清空对话历史\n")

    history: list[dict[str, str]] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit"}:
            print("再见！")
            break
        if user_input.lower() == "clear":
            history.clear()
            print("[对话历史已清空]\n")
            continue

        history.append({"role": "user", "content": user_input})
        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history) + "\nASSISTANT:"

        text, elapsed = generate_fn(engine, prompt, args.max_tokens, args.temperature)
        print(f"Assistant: {text}")
        print(f"[{elapsed:.2f}s]\n")
        history.append({"role": "assistant", "content": text})

    return 0


def main() -> int:
    args = parse_args()
    return chat_loop(build_engine(args), args)


if __name__ == "__main__":
    sys.exit(main())
