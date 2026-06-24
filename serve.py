"""Compatibility entrypoint — 真实实现已移至 mini_infer.cli.serve。

用法同原来完全一致：
  python serve.py --dry-run
  python serve.py --model /path/to/Qwen2.5-7B-Instruct --port 8000

推荐使用 console script：
  mini-infer-serve --dry-run
  mini-infer-serve --model /path/to/Qwen2.5-7B-Instruct
"""
from mini_infer.cli.serve import main, parse_args  # noqa: F401

if __name__ == "__main__":
    main()
