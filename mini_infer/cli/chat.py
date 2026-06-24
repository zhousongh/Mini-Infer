"""mini_infer.cli.chat — console script: mini-infer-chat

快速聊天入口，支持 dry-run（无需模型权重）和真实模型两种模式。

用法：
  mini-infer-chat                                           # dry-run
  mini-infer-chat --real --model-path /path/to/Qwen2.5-7B  # 真实模型
  python quick_chat.py                                      # 同上
"""
from __future__ import annotations

import argparse
import os
import sys

DEFAULT_MODEL_PATH = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"
)


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="mini-infer quick chat",
        add_help=True,
    )
    parser.add_argument("--real", action="store_true", help="使用真实本地模型")
    parser.add_argument("--model-path", type=str, default="", help="本地模型路径")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_known_args(argv)


def resolve_real_model_path(args: argparse.Namespace) -> str:
    path = args.model_path or os.getenv("MODEL") or DEFAULT_MODEL_PATH
    path = os.path.expanduser(path)
    if not os.path.isdir(path):
        raise SystemExit(
            f"找不到本地模型目录：{path}\n"
            "用法：mini-infer-chat --real --model-path /path/to/Qwen2.5-7B-Instruct"
        )
    return path


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args, rest = parse_args(argv)

    from mini_infer.clients.chat_client import main as chat_main

    if args.real or args.model_path:
        chat_argv = [
            "--quick-model-path", resolve_real_model_path(args),
            "--device", args.device,
            "--max-tokens", "64",
            *rest,
        ]
    elif rest:
        chat_argv = rest
    else:
        chat_argv = ["--quick-dry-run", "--max-tokens", "64"]

    return chat_main(chat_argv)


if __name__ == "__main__":
    raise SystemExit(main())
