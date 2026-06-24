"""benchmarks/benchmark_flash_decode.py

Phase 12.5：Flash Decoding（split-K）性能对比 benchmark。

测试对象：
  - reference:    PyTorch 参考实现（baseline）
  - flash_attn:   flash_attn_with_kvcache（工业级对照）
  - triton_65:    Phase 6.5 Triton decode attention kernel（dense KV，标准 grid）
  - flash_decode: Phase 12.5 split-K Triton kernel（dense KV，split-K grid）

指标：
  - 每种 seq_len 下的单步 attention 延迟（ms），warmup=50 次，测量=200 次
  - speedup_vs_flash_attn: flash_decode 相对 flash_attn 的加速比

环境：
  - Ubuntu 24.04 + RTX 4090，批次 batch=1
  - 采用 1.5B 模型的 head 配置（num_q_heads=12, num_kv_heads=2, head_dim=128）
  - 采用 7B 模型的 head 配置可通过 --num-q-heads / --num-kv-heads 调整

注意事项：
  - 仅测 dense KV attention 单步延迟，不含 KV cache 管理和 prefill 开销
  - reference 在大 seq_len 下很慢，可通过 --skip-reference 跳过
"""

import argparse
import time

import torch

# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def make_qkv(batch, seq_len, num_q_heads, num_kv_heads, head_dim, device="cuda"):
    q = torch.randn(batch, 1,       num_q_heads, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=torch.float16, device=device)
    v = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=torch.float16, device=device)
    return q, k, v


def benchmark_fn(fn, warmup=50, repeat=200):
    """在 CUDA 设备上测量函数延迟（ms），先 warmup，再 repeat 次取均值。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / repeat * 1000  # ms


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Flash Decoding split-K benchmark")
    parser.add_argument("--batch", type=int, default=1, help="batch size (default: 1)")
    parser.add_argument("--num-q-heads", type=int, default=12,
                        help="num_q_heads（1.5B=12, 7B=28）")
    parser.add_argument("--num-kv-heads", type=int, default=2,
                        help="num_kv_heads（1.5B=2, 7B=4）")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--skip-reference", action="store_true",
                        help="跳过 PyTorch reference（长 seq_len 时很慢）")
    parser.add_argument("--warmup", type=int, default=50, help="warmup 次数")
    parser.add_argument("--repeat", type=int, default=200, help="测量次数")
    args = parser.parse_args()

    # seq_len sweep
    seq_lens = [128, 256, 512, 1024, 2048, 4096]

    batch       = args.batch
    num_q_heads = args.num_q_heads
    num_kv_heads = args.num_kv_heads
    head_dim    = args.head_dim

    from mini_infer.kernels.triton_attn import (
        flash_decode_attention,
        reference_decode_attention,
        triton_decode_attention,
    )
    from mini_infer.kernels.triton_flash_decode import (
        auto_num_splits,
        flash_decode_triton,
    )

    print(f"\n{'='*80}")
    print(f"Flash Decoding Benchmark — batch={batch}, "
          f"num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}")
    print(f"warmup={args.warmup}, repeat={args.repeat}")
    print(f"{'='*80}")

    # 表头
    cols = ["seq_len", "num_splits"]
    if not args.skip_reference:
        cols += ["ref_ms"]
    cols += ["flash_attn_ms", "triton_65_ms", "flash_decode_ms", "spd_vs_flash", "spd_vs_triton65"]
    print("  ".join(f"{c:>16}" for c in cols))
    print("-" * (18 * len(cols)))

    for seq_len in seq_lens:
        q, k, v = make_qkv(batch, seq_len, num_q_heads, num_kv_heads, head_dim)

        ns = auto_num_splits(seq_len, num_q_heads, batch)

        row = {"seq_len": seq_len, "num_splits": ns}

        if not args.skip_reference:
            ref_ms = benchmark_fn(
                lambda: reference_decode_attention(q, k, v),
                args.warmup, args.repeat,
            )
            row["ref_ms"] = ref_ms

        flash_ms = benchmark_fn(
            lambda: flash_decode_attention(q, k, v),
            args.warmup, args.repeat,
        )
        row["flash_attn_ms"] = flash_ms

        triton65_ms = benchmark_fn(
            lambda: triton_decode_attention(q, k, v),
            args.warmup, args.repeat,
        )
        row["triton_65_ms"] = triton65_ms

        fd_ms = benchmark_fn(
            lambda: flash_decode_triton(q, k, v, num_splits=ns),
            args.warmup, args.repeat,
        )
        row["flash_decode_ms"] = fd_ms

        speedup_vs_flash   = flash_ms / fd_ms
        speedup_vs_triton65 = triton65_ms / fd_ms

        # 打印行
        vals = [f"{seq_len:>16}", f"{ns:>16}"]
        if not args.skip_reference:
            vals.append(f"{row['ref_ms']:>15.3f}")
        vals += [
            f"{flash_ms:>15.3f}",
            f"{triton65_ms:>15.3f}",
            f"{fd_ms:>15.3f}",
            f"{speedup_vs_flash:>14.2f}x",
            f"{speedup_vs_triton65:>16.2f}x",
        ]
        print("  ".join(vals))

    print()
    print("说明：")
    print("  spd_vs_flash    flash_decode 相对 flash_attn_with_kvcache 的加速比")
    print("                  < 1.0 属于正常（flash_attn 是高度优化的 C++ CUDA 实现，Triton 无法匹敌）")
    print("  spd_vs_triton65 flash_decode 相对 Phase 6.5 Triton kernel 的加速比（split-K 的实际收益）")
    print("                  > 1.0 表示 split-K 有益，长序列下预期明显提升")
    print("  flash_decode 延迟在长序列下趋于恒定（SM 利用率提升，不随 seq_len 线性增长）")


if __name__ == "__main__":
    main()
