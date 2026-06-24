"""mini_infer.cli.serve — console script: mini-infer-serve

等价于：python serve.py [args]

用法：
  mini-infer-serve --dry-run --port 8000
  mini-infer-serve --model /path/to/Qwen2.5-7B-Instruct --port 8000
  mini-infer-serve --model /path/to/model --chunk-prefill-size 256 --port 8000
"""
from __future__ import annotations

import argparse

import uvicorn

from mini_infer.core.config import EngineConfig
from mini_infer.serving.server import app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="mini-infer — OpenAI Chat Completions 兼容服务")
    parser.add_argument("--model", type=str, default="", help="模型目录路径")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true", help="使用 stub tokenizer，无需真实模型")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--num-gpu-blocks", type=int, default=200)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--chunk-prefill-size", type=int, default=0,
                        help="每步 prefill 的 token 上限（0=禁用，256 推荐）")
    parser.add_argument("--use-cuda-graph", action="store_true",
                        help="启用 CUDA Graph：decode_batch 静态捕获，降低 Python dispatch 开销")
    parser.add_argument("--quant-mode", type=str, default="",
                        choices=["", "w8a8"],
                        help="量化模式：'' = FP16（默认），'w8a8' = per-channel int8 权重量化")
    parser.add_argument("--scheduler-policy", type=str, default="pressure_aware",
                        choices=["baseline", "reserve_only", "pressure_aware", "adaptive", "slo_aware"],
                        help="调度策略：baseline / reserve_only / pressure_aware / adaptive / slo_aware")
    parser.add_argument("--adaptive-reserve-threshold", type=float, default=0.60,
                        help="adaptive 策略启用 KV 预留的 projected KV utilization 阈值")
    parser.add_argument("--adaptive-preempt-threshold", type=float, default=0.85,
                        help="adaptive 策略启用压力感知抢占的 projected KV utilization 阈值")
    parser.add_argument("--adaptive-waiting-threshold", type=int, default=8,
                        help="adaptive 策略启用保护策略的 waiting queue 长度阈值")
    parser.add_argument("--scheduler-ttft-slo-steps", type=int, default=16,
                        help="slo_aware 策略使用的 TTFT SLO step 阈值")
    parser.add_argument("--scheduler-latency-slo-steps", type=int, default=64,
                        help="slo_aware 策略使用的 latency SLO step 阈值")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.dry_run and not args.model:
        raise SystemExit("请指定 --model 或 --dry-run")

    model_name = args.model if args.model else "dry"
    config = EngineConfig(
        model_name=model_name,
        device=args.device,
        dtype=args.dtype,
        dry_run=args.dry_run,
        max_batch_size=args.max_batch_size,
        num_gpu_blocks=args.num_gpu_blocks,
        block_size=args.block_size,
        chunk_prefill_size=args.chunk_prefill_size,
        use_cuda_graph=args.use_cuda_graph,
        quant_mode=args.quant_mode,
        scheduler_policy=args.scheduler_policy,
        adaptive_reserve_threshold=args.adaptive_reserve_threshold,
        adaptive_preempt_threshold=args.adaptive_preempt_threshold,
        adaptive_waiting_threshold=args.adaptive_waiting_threshold,
        scheduler_ttft_slo_steps=args.scheduler_ttft_slo_steps,
        scheduler_latency_slo_steps=args.scheduler_latency_slo_steps,
    )
    app.state.engine_config = config  # type: ignore[attr-defined]

    print(f"[mini-infer] model={model_name!r}  device={args.device}  port={args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
