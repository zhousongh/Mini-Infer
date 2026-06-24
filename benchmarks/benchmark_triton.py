"""benchmarks/benchmark_triton.py

Phase 6.5：Triton decode attention kernel 性能对比 benchmark。

测试对象：
  - triton_decode_attention   — 本阶段实现的 Triton kernel
  - flash_decode_attention    — flash_attn_with_kvcache（对照组）
  - reference_decode_attention — PyTorch 参考实现（baseline）

测试配置：
  - 模型参数：Qwen2.5-7B（28 Q heads / 4 KV heads / head_dim=128）
  - batch_size：1 和 8
  - seq_len：128 / 512 / 1024 / 2048（decode 阶段典型 context 长度）
  - dtype：float16，CUDA

用法：
  python benchmarks/benchmark_triton.py                    # 跑所有配置
  python benchmarks/benchmark_triton.py --batch_size 1 --seq_len 512
  python benchmarks/benchmark_triton.py --batch_size 8 --seq_len 1024

输出：每个配置的延迟（μs）和相对 flash_attn 的比值，以及 roofline 分析。
"""

import argparse
import sys

import torch
import triton
import triton.testing

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def make_tensors(batch, seq_len, num_q_heads, num_kv_heads, head_dim, device="cuda"):
    q = torch.randn(batch, 1,       num_q_heads,  head_dim, device=device, dtype=torch.float16)
    k = torch.randn(batch, seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    v = torch.randn(batch, seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    return q, k, v


def bench_fn(fn, warmup=25, rep=100):
    """使用 triton.testing.do_bench 测量延迟，返回 μs。

    注意：triton_decode_attention 内部包含 torch.empty_like(q) 的输出分配，
    flash_attn_with_kvcache 内部也分配输出，两者口径一致，比值有效。
    """
    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    return ms * 1e3  # ms → μs


def roofline_analysis(batch, seq_len, num_q_heads, num_kv_heads, head_dim):
    """
    粗略 roofline 分析（RTX 4090）。

    Decode attention 是 memory-bound 操作：
    - Q: batch × num_q_heads × head_dim × 2 bytes
    - K/V: batch × seq_len × num_kv_heads × head_dim × 2 bytes × 2
    - FLOPs: batch × num_q_heads × (2 × seq_len × head_dim + 2 × seq_len + 2 × seq_len × head_dim)
             ≈ batch × num_q_heads × 4 × seq_len × head_dim

    RTX 4090 specs:
    - Peak FP16 TFLOPS: ~82 TFLOPS（tensor core）
    - Peak memory bandwidth: ~1008 GB/s
    - Ridge point: 82e12 / 1008e9 ≈ 81 FLOPs/Byte
    """
    bytes_per_elem = 2  # float16

    # 实际读写的字节数（忽略 softmax 的中间变量）
    bytes_q   = batch * 1 * num_q_heads * head_dim * bytes_per_elem
    bytes_kv  = batch * seq_len * num_kv_heads * head_dim * bytes_per_elem * 2
    bytes_out = batch * 1 * num_q_heads * head_dim * bytes_per_elem
    total_bytes = bytes_q + bytes_kv + bytes_out

    # 浮点运算次数（QK^T + softmax + AV）
    total_flops = batch * num_q_heads * (
        2 * seq_len * head_dim +   # QK^T
        3 * seq_len +              # softmax: exp + sum + div
        2 * seq_len * head_dim     # AV
    )

    arithmetic_intensity = total_flops / total_bytes  # FLOPs/Byte
    ridge_point = 82.0             # RTX 4090, FP16 tensor core
    bw_bound_time_us = total_bytes / (1008e9) * 1e6  # 理论 BW bound 延迟 (μs)

    return {
        "total_bytes_kb":    total_bytes / 1024,
        "total_gflops":      total_flops / 1e9,
        "arith_intensity":   arithmetic_intensity,
        "ridge_point":       ridge_point,
        "is_mem_bound":      arithmetic_intensity < ridge_point,
        "bw_bound_us":       bw_bound_time_us,
    }


# ---------------------------------------------------------------------------
# 主 benchmark 逻辑
# ---------------------------------------------------------------------------

def run_benchmark(batch, seq_len, num_q_heads=28, num_kv_heads=4, head_dim=128):
    from mini_infer.kernels.triton_attn import (
        flash_decode_attention,
        reference_decode_attention,
        triton_decode_attention,
    )

    device = "cuda"
    q, k, v = make_tensors(batch, seq_len, num_q_heads, num_kv_heads, head_dim, device)

    results = {}

    # Triton kernel
    try:
        results["triton"] = bench_fn(lambda: triton_decode_attention(q, k, v))
    except Exception as e:
        results["triton"] = f"ERROR: {e}"

    # flash_attn
    try:
        results["flash"]  = bench_fn(lambda: flash_decode_attention(q, k, v))
    except Exception as e:
        results["flash"]  = f"ERROR: {e}"

    # PyTorch reference（仅 batch=1 时跑，batch=8 太慢意义不大）
    if batch <= 2:
        try:
            results["ref"] = bench_fn(lambda: reference_decode_attention(q, k, v))
        except Exception as e:
            results["ref"] = f"ERROR: {e}"

    # Roofline
    rl = roofline_analysis(batch, seq_len, num_q_heads, num_kv_heads, head_dim)

    print(f"\nbatch={batch}, seq_len={seq_len}, Q_heads={num_q_heads}, KV_heads={num_kv_heads}, head_dim={head_dim}")
    print(f"  Roofline: {rl['total_bytes_kb']:.1f} KB read/write, "
          f"{rl['total_gflops']:.4f} GFLOPS, "
          f"AI={rl['arith_intensity']:.1f} FLOPs/Byte "
          f"({'memory-bound' if rl['is_mem_bound'] else 'compute-bound'}, "
          f"ridge={rl['ridge_point']:.0f})")
    print(f"  Theoretical BW bound: {rl['bw_bound_us']:.2f} μs")

    flash_us = results.get("flash")
    for name, val in results.items():
        if isinstance(val, float):
            ratio = f"  (×{val/flash_us:.2f} vs flash)" if name != "flash" else ""
            print(f"  {name:8s}: {val:8.2f} μs{ratio}")
        else:
            print(f"  {name:8s}: {val}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Triton decode attention benchmark")
    parser.add_argument("--batch_size", type=int,   default=None,  help="指定单一 batch size（默认跑 1 和 8）")
    parser.add_argument("--seq_len",    type=int,   default=None,  help="指定单一 seq_len（默认跑 128/512/1024/2048）")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: 需要 CUDA GPU。当前环境无 GPU，无法运行 benchmark。")
        sys.exit(1)

    import flash_attn
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU:         {device_name}")
    print(f"PyTorch:     {torch.__version__}")
    print(f"Triton:      {triton.__version__}")
    print(f"flash_attn:  {flash_attn.__version__}")
    print("Triton decode attention benchmark（Qwen2.5-7B 参数：28Q/4KV heads, head_dim=128）")
    print("=" * 70)

    batch_sizes = [args.batch_size] if args.batch_size else [1, 8]
    seq_lens    = [args.seq_len]    if args.seq_len    else [128, 512, 1024, 2048]

    for batch in batch_sizes:
        for seq_len in seq_lens:
            run_benchmark(batch, seq_len)

    print("\n" + "=" * 70)
    print("说明：triton/flash 比值 > 1 表示 Triton 比 flash_attn 慢。")
    print("Phase 6.5 目标：triton/flash 比值在合理范围内（阶段目标不是超越 flash_attn，")
    print("而是理解实现路径和 roofline 分析。）")


if __name__ == "__main__":
    main()
