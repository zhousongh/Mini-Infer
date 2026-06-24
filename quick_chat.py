"""Compatibility entrypoint — 真实实现已移至 mini_infer.cli.chat。

用法同原来完全一致：
  python quick_chat.py                # dry-run
  python quick_chat.py --real         # 真实模型

推荐使用 console script：
  mini-infer-chat
  mini-infer-chat --real --model-path /path/to/Qwen2.5-7B-Instruct
"""
from __future__ import annotations
import sys
from mini_infer.cli.chat import main, parse_args, resolve_real_model_path  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
