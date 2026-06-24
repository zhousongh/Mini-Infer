"""mini_infer.cli.demo — console script: mini-infer-demo

对比演示：同一个 prompt，不同配置，并排展示文本输出 + 关键指标。

支持的对比模式：
  quant        FP16 vs W8A8 量化（文本质量 + 显存 + 速度）
  cuda-graph   Eager vs CUDA Graph（decode 延迟）
  prefix-cache 冷启动 vs 前缀命中（TTFT 对比）
  all          依次运行以上三种

用法：
  mini-infer-demo --model /path/to/Qwen2.5-1.5B-Instruct --mode quant
  python demo.py   --model /path/to/Qwen2.5-1.5B-Instruct --mode all
"""
from __future__ import annotations

import argparse
import gc
import textwrap
import time

import torch
from transformers import AutoConfig

from mini_infer.cache.kv_cache import KVCacheManager
from mini_infer.core.config import EngineConfig
from mini_infer.core.request import Request, RequestState, SamplingParams
from mini_infer.modeling.model_runner import ModelRunner
from mini_infer.runtime.engine import LLMEngine

# --------------------------------------------------------------------------- #
DEFAULT_PROMPTS = [
    "The key difference between supervised and unsupervised learning is",
    "To implement a binary search tree in Python, you need to",
]

WIDTH = 72


def _header(title: str) -> None:
    print("\n" + "═" * WIDTH)
    print(f"  {title}")
    print("═" * WIDTH)


def _section(title: str) -> None:
    print(f"\n── {title} " + "─" * (WIDTH - len(title) - 4))


def _wrap(text: str, width: int = 34) -> list[str]:
    lines = textwrap.wrap(text, width=width) or ["(empty)"]
    return lines[:6]


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _infer_geometry(model_path: str) -> dict[str, int]:
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_q = int(cfg.num_attention_heads)
    hidden = int(cfg.hidden_size)
    return {
        "num_hidden_layers": int(cfg.num_hidden_layers),
        "num_kv_heads": int(getattr(cfg, "num_key_value_heads", num_q)),
        "head_dim": int(getattr(cfg, "head_dim", hidden // num_q)),
    }


def _weight_mb(model: torch.nn.Module) -> float:
    total = sum(
        t.numel() * t.element_size()
        for t in list(model.parameters()) + list(model.buffers())
        if t.is_cuda
    )
    return total / 1024 / 1024


def _build_runner(
    model_path: str,
    device: str,
    quant_mode: str = "",
    use_cuda_graph: bool = False,
) -> tuple[EngineConfig, KVCacheManager, ModelRunner]:
    geo = _infer_geometry(model_path)
    cfg = EngineConfig(
        model_name=model_path,
        device=device,
        dtype="float16",
        max_batch_size=4,
        max_model_len=512,
        block_size=256,
        num_gpu_blocks=100,
        quant_mode=quant_mode,
        use_cuda_graph=use_cuda_graph,
        **geo,
    )
    kv = KVCacheManager(cfg)
    runner = ModelRunner(cfg, kv)
    return cfg, kv, runner


def _run_prompts(
    runner: ModelRunner,
    kv: KVCacheManager,
    prompts: list[str],
    max_new_tokens: int = 64,
) -> tuple[list[str], float, float]:
    params = SamplingParams(max_new_tokens=max_new_tokens, temperature=0.0)
    states: list[RequestState] = []
    for i, p in enumerate(prompts):
        token_ids = runner.tokenizer.encode(p, add_special_tokens=True)
        req = Request(request_id=str(i), prompt=p, sampling_params=params)
        st = RequestState(request=req, prompt_token_ids=token_ids)
        states.append(st)
        kv.init_request(st)

    runner.prefill(states)

    active = [s for s in states if not s.finished]
    decode_tokens = 0
    _sync()
    t0 = time.perf_counter()
    for _ in range(max_new_tokens - 1):
        if not active:
            break
        runner.decode_batch(active)
        decode_tokens += len(active)
        active = [s for s in active if not s.finished]
    _sync()
    elapsed = time.perf_counter() - t0

    texts = [
        runner.tokenizer.decode(s.generated_token_ids, skip_special_tokens=True)
        for s in states
    ]
    tps = decode_tokens / elapsed if elapsed > 0 else 0.0
    return texts, tps, _weight_mb(runner.model)


# --------------------------------------------------------------------------- #
# 模式 1：FP16 vs W8A8 量化
# --------------------------------------------------------------------------- #

def demo_quant(model_path: str, device: str, prompts: list[str], max_new_tokens: int) -> None:
    _header("对比模式：FP16 vs W8A8 量化")
    print(f"模型路径 : {model_path}")
    print(f"Prompts  : {len(prompts)} 条，max_new_tokens={max_new_tokens}")

    results: dict[str, dict] = {}
    for mode in ("fp16", "w8a8"):
        _section(f"运行 {mode.upper()} ...")
        _, kv, runner = _build_runner(model_path, device, quant_mode=mode if mode == "w8a8" else "")
        texts, tps, mem = _run_prompts(runner, kv, prompts, max_new_tokens)
        results[mode] = {"texts": texts, "tps": tps, "mem": mem}
        print(f"  显存: {mem:.0f} MB  |  Decode: {tps:.0f} tok/s")
        del runner, kv
        _free_gpu()

    _section("文本输出对比")
    col = 34
    fmt = f"{{:<{col}}}  {{:<{col}}}"
    print(fmt.format("[ FP16 ]", "[ W8A8 ]"))
    print("─" * (col * 2 + 2))
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt[:60]}{'…' if len(prompt) > 60 else ''}")
        fp16_lines = _wrap(results["fp16"]["texts"][i], col)
        w8a8_lines = _wrap(results["w8a8"]["texts"][i], col)
        for j in range(max(len(fp16_lines), len(w8a8_lines))):
            a = fp16_lines[j] if j < len(fp16_lines) else ""
            b = w8a8_lines[j] if j < len(w8a8_lines) else ""
            print(fmt.format(a, b))

    _section("指标汇总")
    fp16, w8a8 = results["fp16"], results["w8a8"]
    mem_ratio = w8a8["mem"] / fp16["mem"]
    tps_ratio = w8a8["tps"] / fp16["tps"] if fp16["tps"] > 0 else float("nan")
    print(f"  {'指标':<20} {'FP16':>10} {'W8A8':>10} {'比值':>8}")
    print(f"  {'-'*52}")
    print(f"  {'显存 (MB)':<20} {fp16['mem']:>10.0f} {w8a8['mem']:>10.0f} {mem_ratio:>7.3f}×")
    print(f"  {'Decode (tok/s)':<20} {fp16['tps']:>10.0f} {w8a8['tps']:>10.0f} {tps_ratio:>7.3f}×")
    print(f"\n  显存节省 {(1-mem_ratio)*100:.1f}%")


# --------------------------------------------------------------------------- #
# 模式 2：Eager vs CUDA Graph
# --------------------------------------------------------------------------- #

def demo_cuda_graph(model_path: str, device: str, prompts: list[str], max_new_tokens: int) -> None:
    _header("对比模式：Eager vs CUDA Graph")
    print(f"模型路径 : {model_path}")

    results: dict[str, dict] = {}
    for use_graph, label in ((False, "eager"), (True, "cuda_graph")):
        _section(f"运行 {label.upper()} ...")
        _, kv, runner = _build_runner(model_path, device, use_cuda_graph=use_graph)
        if use_graph:
            runner.warmup_cuda_graphs()
        texts, tps, mem = _run_prompts(runner, kv, prompts, max_new_tokens)
        results[label] = {"texts": texts, "tps": tps}
        print(f"  Decode: {tps:.0f} tok/s")
        del runner, kv
        _free_gpu()

    _section("正确性验证")
    all_match = True
    for i, (a, b) in enumerate(zip(results["eager"]["texts"], results["cuda_graph"]["texts"])):
        match = a.strip() == b.strip()
        if not match:
            all_match = False
        print(f"  Prompt {i+1}: {'✓ 完全一致' if match else '✗ 输出不同'}")

    _section("指标汇总")
    eager, graph = results["eager"], results["cuda_graph"]
    ratio = graph["tps"] / eager["tps"] if eager["tps"] > 0 else float("nan")
    print(f"  {'模式':<16} {'Decode (tok/s)':>16}")
    print(f"  {'Eager':<16} {eager['tps']:>16.0f}")
    print(f"  {'CUDA Graph':<16} {graph['tps']:>16.0f}")
    direction = "提升" if ratio >= 1 else "退步"
    print(f"\n  CUDA Graph {direction} {abs(ratio-1)*100:.1f}%  |  输出一致性：{'✓ 全部一致' if all_match else '✗ 有差异'}")


# --------------------------------------------------------------------------- #
# 模式 3：Prefix Cache 冷启动 vs 命中
# --------------------------------------------------------------------------- #

def demo_prefix_cache(model_path: str, device: str, max_new_tokens: int) -> None:
    _header("对比模式：Prefix Cache 冷启动 vs 命中")
    geo = _infer_geometry(model_path)
    cfg = EngineConfig(
        model_name=model_path,
        device=device,
        dtype="float16",
        max_batch_size=4,
        max_model_len=1024,
        block_size=256,
        num_gpu_blocks=200,
        **geo,
    )
    engine = LLMEngine(cfg)

    shared_prefix = (
        "You are an expert AI assistant with deep knowledge in computer science, "
        "mathematics, and machine learning. You always give clear, accurate, and "
        "well-structured answers. "
    )
    prompt_a = shared_prefix + "Please explain what a neural network is."
    prompt_b = shared_prefix + "Please explain what gradient descent is."

    _section("第 1 次请求（冷启动，prefix cache miss）")
    _sync()
    t0 = time.perf_counter()
    out_a = engine.generate([prompt_a], max_new_tokens=max_new_tokens)[0]
    _sync()
    ttft_cold = (time.perf_counter() - t0) * 1000
    print(f"  TTFT（冷启动）: {ttft_cold:.1f} ms")
    print(f"  输出: {out_a[:80]}...")

    _section("第 2 次请求（相同前缀，prefix cache hit）")
    _sync()
    t0 = time.perf_counter()
    out_b = engine.generate([prompt_b], max_new_tokens=max_new_tokens)[0]
    _sync()
    ttft_warm = (time.perf_counter() - t0) * 1000
    print(f"  TTFT（命中）  : {ttft_warm:.1f} ms")
    print(f"  输出: {out_b[:80]}...")

    _section("指标汇总")
    reduction = (ttft_cold - ttft_warm) / ttft_cold * 100 if ttft_cold > 0 else 0
    print(f"  {'冷启动（miss）':<20} {ttft_cold:>10.1f} ms")
    print(f"  {'命中（hit）':<20} {ttft_warm:>10.1f} ms")
    print(f"\n  前缀命中 TTFT 减少 {reduction:.1f}%")
    del engine
    _free_gpu()


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="mini-infer 功能对比演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="本地模型目录（Qwen2.5-1.5B 即可）")
    parser.add_argument(
        "--mode",
        default="quant",
        choices=["quant", "cuda-graph", "prefix-cache", "all"],
        help="对比模式（默认: quant）",
    )
    parser.add_argument("--prompt", type=str, default="", help="自定义 prompt（留空使用内置演示）")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS
    modes = ["quant", "cuda-graph", "prefix-cache"] if args.mode == "all" else [args.mode]
    runners = {
        "quant": lambda m, d, p, n: demo_quant(m, d, p, n),
        "cuda-graph": lambda m, d, p, n: demo_cuda_graph(m, d, p, n),
        "prefix-cache": lambda m, d, _p, n: demo_prefix_cache(m, d, n),
    }
    for mode in modes:
        runners[mode](args.model, args.device, prompts, args.max_new_tokens)
        print()


if __name__ == "__main__":
    main()
