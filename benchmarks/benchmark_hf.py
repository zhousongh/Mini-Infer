"""这个文件实现 HuggingFace Transformers baseline benchmark，作为 mini-infer 的对比基准。
测量对象：Qwen2.5 系列模型（或任意 AutoModelForCausalLM 兼容模型）。
指标：TTFT、TPOT、throughput（tokens/s）、peak GPU memory。
运行环境：默认在 Ubuntu 项目环境中执行；如使用 CUDA 设备，需要已就绪的 GPU 和模型权重。
"""

import argparse
import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 固定 prompt 集合，保证 benchmark 可复现
PROMPTS = [
    "请介绍一下大语言模型的推理优化技术。",
    "什么是 PagedAttention？它解决了什么问题？",
    "解释 KV Cache 在 Transformer 推理中的作用。",
    "Continuous Batching 和静态 Batching 的区别是什么？",
    "如何在多 GPU 上做张量并行推理？",
    "介绍一下 FlashAttention 的核心思想。",
    "大模型推理时显存占用的主要来源有哪些？",
    "什么是 Prefill 阶段和 Decode 阶段的区别？",
]


@dataclass
class BenchmarkResult:
    model_name: str
    batch_size: int
    prompt_len: int
    output_len: int
    ttft_ms: float
    tpot_ms: float
    throughput_tok_s: float
    peak_memory_gb: float


def benchmark_hf(
    model_name: str,
    batch_size: int = 4,
    max_new_tokens: int = 128,
    device: str = "cuda:0",
    dtype: str = "float16",
) -> BenchmarkResult:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("benchmark_hf 需要可用的 CUDA GPU，但当前环境未检测到。")

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in dtype_map:
        raise ValueError("dtype 只支持 float16、bfloat16 或 float32")
    torch_dtype = dtype_map[dtype]

    print(f"加载 tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"加载模型: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    prompts = PROMPTS[:batch_size]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    prompt_len = inputs["input_ids"].shape[1]

    print("热身中...")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t_prefill_start = time.perf_counter()
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=1, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ttft_ms = (time.perf_counter() - t_prefill_start) * 1000

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t0

    output_len = out.shape[1] - inputs["input_ids"].shape[1]
    total_tokens = output_len * batch_size
    tpot_ms = (total_time * 1000 - ttft_ms) / max(output_len - 1, 1)
    throughput = total_tokens / total_time
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    return BenchmarkResult(
        model_name=model_name,
        batch_size=batch_size,
        prompt_len=prompt_len,
        output_len=output_len,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        throughput_tok_s=throughput,
        peak_memory_gb=peak_mem_gb,
    )


def print_result(result: BenchmarkResult) -> None:
    print("\n========== HuggingFace Baseline Benchmark ==========")
    print(f"模型:       {result.model_name}")
    print(f"batch_size: {result.batch_size}")
    print(f"prompt_len: {result.prompt_len} tokens (padded)")
    print(f"output_len: {result.output_len} tokens")
    print(f"TTFT:       {result.ttft_ms:.1f} ms")
    print(f"TPOT:       {result.tpot_ms:.2f} ms/token")
    print(f"Throughput: {result.throughput_tok_s:.1f} tokens/s")
    print(f"Peak Mem:   {result.peak_memory_gb:.2f} GB")
    print("=====================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="HuggingFace baseline benchmark")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="float16")
    args = parser.parse_args()

    result = benchmark_hf(
        model_name=args.model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
    )
    print_result(result)


if __name__ == "__main__":
    main()
