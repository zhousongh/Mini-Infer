"""Compatibility entrypoint — 真实实现已移至 mini_infer.cli.demo。

用法同原来完全一致：
  python demo.py --model /path/to/Qwen2.5-1.5B-Instruct --mode quant
  python demo.py --model /path/to/Qwen2.5-1.5B-Instruct --mode all

推荐使用 console script：
  mini-infer-demo --model /path/to/Qwen2.5-1.5B-Instruct --mode all
"""
from mini_infer.cli.demo import main

if __name__ == "__main__":
    main()
